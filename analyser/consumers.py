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
FRAMES_PER_CLIP = 17  # Align with main.py
CHECKPOINT_PATH = 'models/best_model.pth'

# Transform for images (match JesterSequenceDataset)
transform = transforms.Compose([
    transforms.Resize((100, 100), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Actions list
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
        self.processing_task = None
        self.max_frames = FRAMES_PER_CLIP * 2  # 17 * 2 = 34 (frames + keypoints)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model from {CHECKPOINT_PATH} on device {self.device}")
        start_time = time.time()
        self.model = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=False)
        self.model.eval()
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")

    async def connect(self):
        await self.accept()
        print("WebSocket connected")
        await self.send(text_data=json.dumps({'message': 'WebSocket connected'}))

    async def disconnect(self, close_code):
        print(f"WebSocket disconnected with code: {close_code}")
        if self.processing_task is not None:
            self.processing_task.cancel()
        await self.send(text_data=json.dumps({'message': 'WebSocket disconnected'}))

    async def receive(self, text_data=None, bytes_data=None):
        print(f"Received data - text_data: {text_data[:30] if text_data else None}, bytes_data: {bytes_data[:30] if bytes_data else None}")
        if text_data:
            try:
                start_time = time.time()
                # Decode base64 string from text_data
                image_data = base64.b64decode(text_data)
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                decode_time = time.time() - start_time
                print(f"Image decoded in {decode_time:.2f} seconds")

                # Transform image
                start_time = time.time()
                image_tensor = transform(image).to(self.device)
                transform_time = time.time() - start_time
                print(f"Image transformed in {transform_time:.2f} seconds")
                self.frame_queue.append(image_tensor)

                # Record time of last frame
                self.last_frame_time = time.time()

                # Process keypoints asynchronously
                start_time = time.time()
                keypoints = await sync_to_async(self.process_keypoints)(image)
                keypoint_time = time.time() - start_time
                print(f"Keypoints extracted in {keypoint_time:.2f} seconds")
                self.frame_queue.append(keypoints)

                if len(self.frame_queue) >= self.max_frames:
                    # Extract frames and keypoints
                    clip_data = self.frame_queue[:self.max_frames]
                    frames = clip_data[0::2]  # Image tensors (0, 2, 4, ...)
                    keypoints_list = clip_data[1::2]  # Keypoints (1, 3, 5, ...)
                    # Remove processed elements from queue
                    self.frame_queue = self.frame_queue[self.max_frames:]
                    # Cancel any ongoing task
                    if self.processing_task is not None and not self.processing_task.done():
                        self.processing_task.cancel()
                        try:
                            await self.processing_task
                        except asyncio.CancelledError:
                            pass
                    print("Starting clip processing task")
                    self.processing_task = asyncio.create_task(self.process_clip(frames, keypoints_list))
            except Exception as e:
                print(f"Error processing image: {e}")
                await self.send(text_data=json.dumps({'error': f'Image processing failed: {str(e)}'}))
        else:
            print("No text_data received")

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
        return torch.tensor(keypoints, dtype=torch.float32).to(self.device)

    async def process_clip(self, frames, keypoints_list):
        """Process a clip of frames and keypoints through the model."""
        print("Processing clip")
        start_time = time.time()

        try:
            # Stack frames and keypoints
            clip = torch.stack(frames).unsqueeze(0)  # [1, T, C, H, W]
            clip = clip.permute(0, 2, 1, 3, 4)  # [1, C, T, H, W]
            joint_stream = torch.stack(keypoints_list).unsqueeze(0)  # [1, T, 48, 3]
            print(f"Clip shape: {clip.shape}, Joint stream shape: {joint_stream.shape}")

            # Run inference
            with torch.no_grad():
                inference_start = time.time()
                outputs = self.model(clip, joint_stream)
                inference_time = time.time() - inference_start
                print(f"Inference completed in {inference_time:.2f} seconds")

            # Handle model output (tensor, not dictionary)
            if isinstance(outputs, dict):
                final_output = outputs['final_output']  # For compatibility with dict output
            else:
                final_output = outputs  # Direct tensor output [1, 18]

            probs = torch.softmax(final_output, dim=1).squeeze(0).cpu().numpy()
            pred_class = probs.argmax().item()
            pred_label = ACTIONS[pred_class]
            confidence = probs[pred_class].item()
            print(f"Prediction: {pred_label} with confidence {confidence:.2f}")

            # Debug info
            debug_info = {}
            if isinstance(outputs, dict):
                for key, output in outputs.items():
                    if torch.isnan(output).any() or torch.isinf(output).any():
                        debug_info[key] = "NaN or Inf detected"
                    else:
                        debug_info[key] = str(output.shape)
            else:
                debug_info['output'] = str(outputs.shape) if not (torch.isnan(outputs).any() or torch.isinf(outputs).any()) else "NaN or Inf detected"

            # Calculate latency
            latency = time.time() - self.last_frame_time
            print(f"Total clip processing latency: {latency:.2f} seconds")

            # Send result
            result = {
                'prediction': pred_label,
                'confidence': float(confidence),
                'latency': latency,
                'debug_info': debug_info
            }
            print(f"Sending result: {result}")
            await self.send(text_data=json.dumps(result))
        except Exception as e:
            print(f"Inference failed: {e}")
            await self.send(text_data=json.dumps({'error': f'Inference failed: {str(e)}'}))