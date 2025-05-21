
import json
import base64
import time
import numpy as np
import torch
from PIL import Image
import io
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings
from asgiref.sync import sync_to_async
import mediapipe as mp
from torchvision import transforms

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Model parameters
NUM_CLASSES = 18
JOINT_DIM = 144
NUM_JOINTS = 48
COORDS = 3
FRAMES_PER_CLIP = 12
CHECKPOINT_PATH = 'models/best_model.pth'

# Transform for images (match JesterSequenceDataset)
transform = transforms.Compose([
    transforms.Resize((100, 100), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Actions list (replace with your actual actions)
ACTIONS = [
    "Doing other things", "No gesture", "Rolling Hand Backward", "Rolling Hand Forward",
    "Shaking Hand", "Sliding Two Fingers Down", "Sliding Two Fingers Left",
    "Sliding Two Fingers Right", "Sliding Two Fingers Up", "Stop Sign",
    "Swiping Down", "Swiping Left", "Swiping Right", "Swiping Up",
    "Thumb Down", "Thumb Up", "Turning Hand Clockwise", "Turning Hand Counterclockwise"
]
label2id = {action: idx for idx, action in enumerate(ACTIONS)}


def extract_keypoints(pose_landmarks, left_hand_landmarks, right_hand_landmarks):
    """Extract keypoints from pose and hand landmarks."""
    keypoints = []
    pose_indices = [11, 12, 13, 14, 15, 16]  # Shoulders, elbows, wrists
    if pose_landmarks:
        for idx in pose_indices:
            lm = pose_landmarks.landmark[idx]
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0.0] * 3 * len(pose_indices))

    for hand_landmarks in [left_hand_landmarks, right_hand_landmarks]:
        if hand_landmarks:
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 3 * 21)
    return np.array(keypoints).reshape(48, 3)

class GestureConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_queue = []
        self.last_frame_time = None
        self.processing_task = None  # Initialize processing_task
        self.max_frames = FRAMES_PER_CLIP * 2  # Define max_frames (e.g., 24)
        # Load model (do this once per consumer instance)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=False)
        self.model.eval()

    async def connect(self):
        await self.accept()
        await self.send(text_data=json.dumps({'message': 'WebSocket connected'}))

    async def disconnect(self, close_code):
        print(f"WebSocket disconnected with code: {close_code}")
        # Cancel any ongoing processing task
        if self.processing_task is not None:
            self.processing_task.cancel()
        await self.send(text_data=json.dumps({'message': 'WebSocket disconnected'}))
        # Note: pose and hands are global, so closing them here may not be necessary
        # pose.close()
        # hands.close()

    async def receive(self, text_data=None, bytes_data=None):
        print("Received data")
        if bytes_data:
            try:
                # Decode base64 image
                image_data = base64.b64decode(bytes_data)
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
            except Exception as e:
                print(f"Error decoding image: {e}")
                await self.send(text_data=json.dumps({'error': f'Image decoding failed: {str(e)}'}))
                return

            # Transform image
            image_tensor = transform(image).to(self.device)  # [C, H, W]
            self.frame_queue.append(image_tensor)

            # Record time of last frame
            self.last_frame_time = time.time()

            # Process keypoints asynchronously
            keypoints = await sync_to_async(self.process_keypoints)(image)
            self.frame_queue.append(keypoints)

            # Check if we have enough frames
            if len(self.frame_queue) >= self.max_frames:
                # Cancel any ongoing processing task
                if self.processing_task is not None and not self.processing_task.done():
                    self.processing_task.cancel()
                    try:
                        await self.processing_task  # Wait for cancellation
                    except asyncio.CancelledError:
                        pass

                # Start new processing task
                self.processing_task = asyncio.create_task(self.process_clip())
                # Reset queue for new frames
                self.frame_queue = []

    def process_keypoints(self, image):
        """Extract keypoints using MediaPipe."""
        image_np = np.array(image)
        pose_results = pose.process(image_np)
        hands_results = hands.process(image_np)

        left_hand = hands_results.multi_hand_landmarks[0] if hands_results.multi_hand_landmarks else None
        right_hand = hands_results.multi_hand_landmarks[1] if hands_results.multi_hand_landmarks and len(hands_results.multi_hand_landmarks) > 1 else None
        keypoints = extract_keypoints(
            pose_results.pose_landmarks if pose_results.pose_landmarks else None,
            left_hand, right_hand
        )
        return torch.tensor(keypoints, dtype=torch.float32).to(self.device)  # [48, 3]

    async def process_clip(self):
        """Process a clip of frames and keypoints through the model."""
        start_time = time.time()

        # Extract frames and keypoints
        frames = self.frame_queue[:FRAMES_PER_CLIP]
        keypoints = self.frame_queue[FRAMES_PER_CLIP:FRAMES_PER_CLIP * 2]
        # Note: Queue is cleared in receive after task creation

        # Prepare clip
        clip = torch.stack(frames).unsqueeze(0)  # [1, T, C, H, W]
        clip = clip.permute(0, 2, 1, 3, 4)  # [1, C, T, H, W]
        joint_stream = torch.stack(keypoints).unsqueeze(0)  # [1, T, 48, 3]

        # Run inference
        try:
            with torch.no_grad():
                outputs = self.model(clip, joint_stream)

            # Extract final prediction
            final_output = outputs['final_output']  # [1, 18]
            probs = torch.softmax(final_output, dim=1).squeeze(0).cpu().numpy()
            pred_class = probs.argmax().item()
            pred_label = ACTIONS[pred_class]
            confidence = probs[pred_class].item()

            # Debug info
            debug_info = {}
            for key, output in outputs.items():
                if torch.isnan(output).any() or torch.isinf(output).any():
                    debug_info[key] = "NaN or Inf detected"
                else:
                    debug_info[key] = str(output.shape)

            # Calculate latency
            latency = time.time() - self.last_frame_time

            # Send result
            result = {
                'prediction': pred_label,
                'confidence': float(confidence),
                'latency': latency,
                'debug_info': debug_info
            }
            await self.send(text_data=json.dumps(result))

        except Exception as e:
            await self.send(text_data=json.dumps({'error': f'Inference failed: {str(e)}'}))