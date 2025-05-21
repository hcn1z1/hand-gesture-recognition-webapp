import torch
import torch.nn as nn

class C3DImproved(nn.Module):
    def __init__(self, num_classes=18, joint_dim=144, lstm_hidden_size=512, num_joints=48, coords=3):
        super().__init__()
        # 3D CNN to process video clips
        self.cnn3d = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )
        self.cnn_out_channels = 256
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(self.cnn_out_channels + joint_dim, lstm_hidden_size, batch_first=True)
        # Classification layers
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
        self.joint_dim = joint_dim
        self.num_joints = num_joints
        self.coords = coords

    def forward(self, clip, joint_stream=None):
        # clip shape: [B, C, T, H, W]
        B, C, T, H, W = clip.size()
        # Process video clip through 3D CNN
        x = self.cnn3d(clip)  # [B, 256, T, H', W']
        x = x.mean(dim=(3, 4))  # Global average pooling over spatial dims: [B, 256, T]
        x = x.permute(0, 2, 1)  # [B, T, 256]
        
        # Handle optional joint stream
        if joint_stream is None:
            joint_stream = torch.zeros(B, T, self.joint_dim, device=clip.device)
        else:
            # Expect joint_stream as [B, T, num_joints, coords] and reshape to [B, T, joint_dim]
            if joint_stream.dim() == 4:
                assert joint_stream.shape == (B, T, self.num_joints, self.coords), \
                    f"Expected joint_stream shape {(B, T, self.num_joints, self.coords)}, got {joint_stream.shape}"
                joint_stream = joint_stream.view(B, T, self.num_joints * self.coords)
            assert joint_stream.shape == (B, T, self.joint_dim), \
                f"Expected joint_stream shape {(B, T, self.joint_dim)}, got {joint_stream.shape}"
        
        # Concatenate CNN features with joint stream
        x = torch.cat([x, joint_stream], dim=2)  # [B, T, 256 + joint_dim]
        # Process sequence through LSTM
        x, _ = self.lstm(x)  # [B, T, 512]
        x = self.dropout(x[:, -1, :])  # Last time step: [B, 512]
        # Classify
        x = self.fc(x)  # [B, num_classes]
        return x