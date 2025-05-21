import json
import base64
import time
import numpy as np
import torch
from PIL import Image
import io
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
import mediapipe as mp
from torchvision import transforms

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=0)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Model settings
NUM_CLASSES = 18
FRAMES_PER_CLIP = 12  # 12 frames per clip
CHECKPOINT_PATH = 'models/best_model.pth'
ACTIONS = [
    "Doing other things", "No gesture", "Rolling Hand Backward", "Rolling Hand Forward",
    "Shaking Hand", "Sliding Two Fingers Down", "Sliding Two Fingers Left",
    "Sliding Two Fingers Right", "Sliding Two Fingers Up", "Stop Sign",
    "Swiping Down", "Swiping Left", "Swiping Right", "Swiping Up",
    "Thumb Down", "Thumb Up", "Turning Hand Clockwise", "Turning Hand Counterclockwise"
]

# Image transformation
transform = transforms.Compose([
    transforms.Resize((100, 100), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def extract_keypoints(pose_landmarks, left_hand_landmarks, right_hand_landmarks):
    """Extract keypoints from pose and hands."""
    keypoints = []
    pose_indices = [11, 12, 13, 14, 15, 16]  # Shoulders, elbows, wrists
    if pose_landmarks:
        for idx in pose_indices:
            lm = pose_landmarks.landmark[idx]
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0.0] * 18)  # 6 joints * 3 coords

    for hand_landmarks in [left_hand_landmarks, right_hand_landmarks]:
        if hand_landmarks:
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 63)  # 21 joints * 3 coords
    return np.array(keypoints).reshape(48, 3)

class GestureConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_queue = []  # Queue for frames and keypoints
        self.processing_lock = asyncio.Lock()  # Lock to block during processing
        self.max_frames = FRAMES_PER_CLIP * 2  # 24 (12 frames + 12 keypoints)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=False)
        self.model.eval()
        print(f"Model loaded on {self.device}")

    async def connect(self):
        await self.accept()
        print("WebSocket connected")

    async def disconnect(self, close_code):
        self.frame_queue = []  # Clear queue on disconnect
        print(f"WebSocket disconnected: {close_code}")

    async def receive(self, text_data=None, bytes_data=None):
        if text_data:
            # If processing is happening, ignore the frame
            if self.processing_lock.locked():
                print("Processing in progress, skipping frame")
                return

            # Lock the processing so no new frames sneak in
            async with self.processing_lock:
                # Decode the image
                image_data = base64.b64decode(text_data)
                image = Image.open(io.BytesIO(image_data)).convert('RGB')

                # Transform image to tensor
                image_tensor = transform(image).to(self.device)
                self.frame_queue.append(image_tensor)

                # Extract keypoints
                keypoints = await sync_to_async(self.process_keypoints)(image)
                self.frame_queue.append(keypoints)

                # Check if queue is full
                if len(self.frame_queue) >= self.max_frames:
                    # Split into frames and keypoints
                    frames = self.frame_queue[0::2]  # Every other element starting at 0
                    keypoints_list = self.frame_queue[1::2]  # Every other starting at 1
                    print("Queue full, processing clip")
                    await self.process_clip(frames, keypoints_list)
                    # Clear the queue after processing
                    self.frame_queue = []
                    print("Queue emptied, ready for new frames")

    def process_keypoints(self, image):
        """Extract keypoints from the image."""
        image_np = np.array(image)
        pose_results = pose.process(image_np)
        hands_results = hands.process(image_np)
        left_hand = hands_results.multi_hand_landmarks[0] if hands_results.multi_hand_landmarks else None
        right_hand = hands_results.multi_hand_landmarks[1] if hands_results.multi_hand_landmarks and len(hands_results.multi_hand_landmarks) > 1 else None
        keypoints = extract_keypoints(
            pose_results.pose_landmarks if pose_results.pose_landmarks else None,
            left_hand, right_hand
        )
        return torch.tensor(keypoints, dtype=torch.float32).to(self.device)

    async def process_clip(self, frames, keypoints_list):
        """Process the clip and send the prediction."""
        try:
            # Prepare the data
            clip = torch.stack(frames).unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, C, T, H, W]
            joint_stream = torch.stack(keypoints_list).unsqueeze(0)  # [1, T, 48, 3]

            # Run the model
            with torch.no_grad():
                outputs = self.model(clip, joint_stream)
                probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
                pred_class = probs.argmax().item()
                pred_label = ACTIONS[pred_class]
                confidence = probs[pred_class].item()

            # Send the result
            result = {
                'type': 'info',
                'prediction': pred_label,
                'confidence': float(confidence)
            }
            await self.send(text_data=json.dumps(result))
            print(f"Predicted: {pred_label} with confidence {confidence:.2f}")
        except Exception as e:
            print(f"Error processing clip: {e}")
            await self.send(text_data=json.dumps({'type': 'info', 'error': str(e)}))
