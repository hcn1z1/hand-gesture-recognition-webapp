import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import threading
import queue
import time

# ------------------------------------------------------------------
# 1) Define (or import) your 3D CNN model
# ------------------------------------------------------------------

class MultiScaleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[0], kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_channels[0]), nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[1], kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels[1]), nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[2], kernel_size=5, stride=stride, padding=2),
            nn.BatchNorm2d(out_channels[2]), nn.ReLU()
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return torch.cat([b1, b2, b3], dim=1)

class ImprovedGestureModel(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.ms1 = MultiScaleLayer(64, [32, 32, 32])
        self.pool1 = nn.AvgPool2d(2, 2)
        self.ms2 = MultiScaleLayer(96, [48, 48, 48], stride=2)
        self.fc_reduce = nn.Linear(15552, 48)  # Reduce CNN output to 48 features
        self.lstm = nn.LSTM(192, 256, batch_first=True)  # 48 (CNN) + 144 (joint) = 192
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, clip, joint_stream):
        batch_size, c, seq_len, h, w = clip.size()
        x = clip.contiguous().reshape(batch_size * seq_len, c, h, w)  # [B*T, C, H, W]
        x = self.relu(self.conv1(x))
        x = self.ms1(x)
        x = self.pool1(x)
        x = self.ms2(x)
        # Reshape and reduce dimensions
        x = x.view(batch_size, seq_len, -1)  # [B, T, C*H*W]
        x = self.fc_reduce(x)  # [B, T, 48]
        # Process joint stream
        x_joint = joint_stream.view(batch_size, seq_len, -1)  # [B, T, 144]
        x = torch.cat([x, x_joint], dim=2)  # [B, T, 192]
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x

class C3DGestureLSTM(nn.Module):
    def __init__(self, num_classes=18):
        super(C3DGestureLSTM, self).__init__()
        # 3D CNN layers
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.conv4 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1)
        self.conv5 = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=1)
        self.conv6 = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=1)
        self.global_pool = nn.AdaptiveMaxPool3d((12, 1, 1))  # Preserve 12 frames
        # LSTM layers
        self.lstm1 = nn.LSTM(256, 256, batch_first=True)
        self.lstm2 = nn.LSTM(256, 256, batch_first=True)
        # Fully connected layers
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, clip, joint_stream=None):
        # clip: [B, C, T, H, W] (e.g., [B, 3, 12, 100, 100])
        # joint_stream: [B, T, 48, 3] (optional, for future extension)
        batch_size, channels, seq_len, height, width = clip.size()

        # Video branch (3D CNN)
        x = self.relu(self.conv1(clip))  # [B, 32, T, H/2, W/2]
        x = self.pool1(x)                # [B, 32, T, H/4, W/4]
        x = self.relu(self.conv2(x))     # [B, 64, T, H/4, W/4]
        x = self.pool2(x)                # [B, 64, T, H/8, W/8]
        x = self.relu(self.conv3(x))     # [B, 128, T, H/8, W/8]
        x = self.pool3(x)                # [B, 128, T, H/16, W/16]
        x = self.relu(self.conv4(x))     # [B, 256, T, H/16, W/16]
        x = self.relu(self.conv5(x))     # [B, 256, T, H/16, W/16]
        x = self.relu(self.conv6(x))     # [B, 256, T, H/16, W/16]
        x = self.global_pool(x)          # [B, 256, T, 1, 1]
        x_video = x.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # [B, T, 256]

        # LSTM processing
        x, _ = self.lstm1(x_video)       # [B, T, 256]
        x, _ = self.lstm2(x)             # [B, T, 256]
        x = x[:, -1, :]                  # [B, 256] (take last time step)

        # Fully connected layers
        x = self.relu(self.fc1(x))       # [B, 256]
        x = self.fc2(x)                  # [B, num_classes]
        return x
    


# ------------------------------------------------------------------
# 2) Model loading
# ------------------------------------------------------------------
def load_model(path, device):
    model = C3DGestureLSTM(num_classes=18)
    state_dict = torch.load(path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Missing keys:    {missing}")
    print(f"  Unexpected keys: {unexpected}")
    model.to(device)
    model.eval()
    return model

# ------------------------------------------------------------------
# 3) Frame preprocessing (100×100 here; adjust if needed)
# ------------------------------------------------------------------
def preprocess_frame(frame):
    # BGR -> RGB PIL, resize, normalize
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = img.resize((100, 100), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0     # H×W×3
    arr = arr.transpose(2, 0, 1)                      # 3×H×W
    t   = torch.from_numpy(arr)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return (t - mean) / std

# ------------------------------------------------------------------
# 4) Threaded processing: collect 17 frames → (1,3,17,100,100)
# ------------------------------------------------------------------
def process_frames(model, frame_queue, device):
    while True:
        frames = []
        while len(frames) < 17:
            try:
                # get a preprocessed [3,100,100] tensor
                frames.append(frame_queue.get(timeout=1).to(device))
            except queue.Empty:
                continue

        # Stack into [17,3,100,100], then rearrange to [1,3,17,100,100]
        clip = torch.stack(frames)            # (T, C, H, W)
        clip = clip.permute(1, 0, 2, 3)       # (C, T, H, W)
        clip = clip.unsqueeze(0)             # (1, C, T, H, W)

        with torch.no_grad():
            logits = model(clip)              # (1, num_classes)
            _, pred = torch.max(logits, dim=1)
        print(f"[{time.strftime('%H:%M:%S')}] Predicted class: {pred.item()}")

        # clear any leftovers
        while not frame_queue.empty():
            frame_queue.get()

# ------------------------------------------------------------------
# 5) Main: open video, push frames, start processing thread
# ------------------------------------------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = load_model('models/c3d_best_model.pth', device)

    frame_q = queue.Queue(maxsize=17)
    threading.Thread(
        target=process_frames,
        args=(model, frame_q, device),
        daemon=True
    ).start()

    cap = cv2.VideoCapture('test/test.mp4')
    if not cap.isOpened():
        print("Error: could not open video file.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Reached end of video.")
                break

            if not frame_q.full():
                tensor = preprocess_frame(frame)
                frame_q.put(tensor)
                print(f"Queued frame ({frame_q.qsize()}/17)")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User requested exit.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
