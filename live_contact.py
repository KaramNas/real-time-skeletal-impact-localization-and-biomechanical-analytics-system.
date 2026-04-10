import cv2
import torch
import numpy as np
import sys
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from torchvision import transforms

# ─── Path Setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, 'HOT/HOT-main')
from hot.config import cfg
from hot.models import ModelBuilder, SegmentationModule
import torch.nn as nn

# ─── Config ───────────────────────────────────────────────────────────────────
ENCODER_WEIGHTS = 'HOT/HOT-main/ckpt/hot-c1/encoder_epoch_14.pth'
DECODER_WEIGHTS = 'HOT/HOT-main/ckpt/hot-c1/decoder_epoch_14.pth'
CONFIG_FILE     = 'HOT/HOT-main/config/hot-resnet50dilated-c1.yaml'
MODEL_PATH      = 'pose_landmarker_lite.task'
CAMERA_INDEX    = 0
DISPLAY_WIDTH   = 1280
DISPLAY_HEIGHT  = 720

# ─── Contact class colors (18 classes) ────────────────────────────────────────
CONTACT_COLORS = np.array([
    [0,   0,   0  ],  # 0  background - transparent
    [255, 0,   0  ],  # 1  red
    [0,   255, 0  ],  # 2  green
    [0,   0,   255],  # 3  blue
    [255, 255, 0  ],  # 4  yellow
    [255, 0,   255],  # 5  magenta
    [0,   255, 255],  # 6  cyan
    [255, 128, 0  ],  # 7  orange
    [128, 0,   255],  # 8  purple
    [0,   128, 255],  # 9  light blue
    [255, 0,   128],  # 10 pink
    [0,   255, 128],  # 11 mint
    [128, 255, 0  ],  # 12 lime
    [255, 128, 128],  # 13 salmon
    [128, 128, 255],  # 14 lavender
    [128, 255, 128],  # 15 light green
    [255, 200, 0  ],  # 16 gold
    [0,   200, 255],  # 17 sky blue
], dtype=np.uint8)

# ─── Pose Connections ─────────────────────────────────────────────────────────
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (15,17),(15,19),(15,21),(16,18),(16,20),(16,22),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(29,31),
    (24,26),(26,28),(28,30),(30,32)
]

# ─── Load HOT Model ───────────────────────────────────────────────────────────
print("Loading HOT model...")
cfg.merge_from_file(CONFIG_FILE)

net_encoder = ModelBuilder.build_encoder(
    arch=cfg.MODEL.arch_encoder.lower(),
    fc_dim=cfg.MODEL.fc_dim,
    weights=ENCODER_WEIGHTS)

net_decoder = ModelBuilder.build_decoder(
    cfg=cfg,
    arch=cfg.MODEL.arch_decoder.lower(),
    fc_dim=cfg.MODEL.fc_dim,
    num_class=cfg.DATASET.num_class,
    weights=DECODER_WEIGHTS)

crit = nn.NLLLoss()
segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
segmentation_module.cuda()
segmentation_module.eval()
print("HOT model loaded ✓")

# ─── Load MediaPipe ───────────────────────────────────────────────────────────
print("Loading MediaPipe...")
BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1)

landmarker = PoseLandmarker.create_from_options(options)
print("MediaPipe loaded ✓")

# ─── Image Preprocessing for HOT ─────────────────────────────────────────────
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

def preprocess_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (400, 300))
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)
    img_tensor = normalize(img_tensor)
    img_tensor = img_tensor.unsqueeze(0).cuda()
    return img_tensor

# ─── Run HOT Inference ────────────────────────────────────────────────────────
def run_hot(frame):
    img_tensor = preprocess_frame(frame)
    feed_dict = {'img_data': img_tensor}
    with torch.no_grad():
        scores, _, _ = segmentation_module.decoder(
            segmentation_module.encoder(img_tensor, return_feature_maps=True))
    pred = torch.argmax(scores, dim=1).squeeze(0).cpu().numpy()
    return pred

# ─── Build Contact Overlay ────────────────────────────────────────────────────
def build_overlay(pred, frame_shape):
    h, w = frame_shape[:2]
    color_map = CONTACT_COLORS[pred]
    overlay = cv2.resize(color_map, (w, h), interpolation=cv2.INTER_NEAREST)
    return overlay

# ─── Draw Skeleton ────────────────────────────────────────────────────────────
def draw_skeleton(frame, result):
    if not result.pose_landmarks:
        return frame
    for pose_landmarks in result.pose_landmarks:
        points = []
        for landmark in pose_landmarks:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 255), 2)
    return frame

# ─── Main Loop ────────────────────────────────────────────────────────────────
print("Starting camera... Press Q to quit")
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)

frame_timestamp_ms = 0
fps_counter = 0
fps_display = 0
fps_timer = cv2.getTickCount()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ── HOT Contact Detection ──
    pred = run_hot(frame)
    overlay = build_overlay(pred, frame.shape)

    # ── Blend contact overlay with original frame ──
    contact_mask = (pred > 0)  # non-background
    contact_mask_resized = cv2.resize(
        contact_mask.astype(np.uint8),
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_NEAREST).astype(bool)

    blended = frame.copy()
    blended[contact_mask_resized] = cv2.addWeighted(
        frame, 0.4,
        overlay, 0.6, 0)[contact_mask_resized]

    # ── MediaPipe Skeleton ──
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    pose_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    blended = draw_skeleton(blended, pose_result)

    # ── FPS Counter ──
    fps_counter += 1
    elapsed = (cv2.getTickCount() - fps_timer) / cv2.getTickFrequency()
    if elapsed >= 1.0:
        fps_display = fps_counter
        fps_counter = 0
        fps_timer = cv2.getTickCount()

    cv2.putText(blended, f"FPS: {fps_display}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(blended, "Contact Detection + Skeleton", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Live Human-Object Contact Detection", blended)

    frame_timestamp_ms += 33
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Session ended.")