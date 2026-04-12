import cv2
import torch
import numpy as np
import mediapipe as mp
import sys, os
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

# ── Paths ────────────────────────────────────────────────────────────────────
HOT_ROOT = os.path.join(os.path.dirname(__file__), 'HOT', 'HOT-main')
sys.path.insert(0, HOT_ROOT)

from hot.models import ModelBuilder, SegmentationModule
from hot.config import cfg

CFG_FILE  = os.path.join(HOT_ROOT, 'config', 'hot-resnet50dilated-c1.yaml')
CKPT_DIR  = os.path.join(HOT_ROOT, 'ckpt', 'hot-c1')
THRESHOLD = 0.7

# ── Load HOT model ───────────────────────────────────────────────────────────
cfg.merge_from_file(CFG_FILE)
cfg.freeze()

builder     = ModelBuilder()
net_encoder = builder.build_encoder(
    arch=cfg.MODEL.arch_encoder,
    fc_dim=cfg.MODEL.fc_dim,
    weights=os.path.join(CKPT_DIR, 'encoder_epoch_14.pth'))
net_decoder = builder.build_decoder(
    arch=cfg.MODEL.arch_decoder,
    fc_dim=cfg.MODEL.fc_dim,
    num_class=cfg.DATASET.num_class,
    cfg=cfg,
    weights=os.path.join(CKPT_DIR, 'decoder_epoch_14.pth'))

seg_module = SegmentationModule(net_encoder, net_decoder, None)
seg_module.cuda().eval()

# ── Load YOLO ────────────────────────────────────────────────────────────────
yolo = YOLO("yolov8n.pt")

# ── Load MediaPipe ───────────────────────────────────────────────────────────
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions           = python.BaseOptions
PoseLandmarker        = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode     = vision.RunningMode

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_lite.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1)
landmarker = PoseLandmarker.create_from_options(pose_options)

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (15,17),(15,19),(15,21),(16,18),(16,20),(16,22),
    (11,23),(12,24),(23,24),(23,25),(25,27),(27,29),(29,31),
    (24,26),(26,28),(28,30),(30,32)
]

# ── HOT preprocessing ────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

CONTACT_COLOR = np.array([0, 0, 255], dtype=np.uint8)

# ── Frame skip settings ──────────────────────────────────────────────────────
YOLO_EVERY = 10   # run YOLO every N frames
HOT_EVERY  = 3    # run HOT every N frames

# ── Helper functions ─────────────────────────────────────────────────────────
def run_hot(frame):
    h, w = frame.shape[:2]
    img_pil     = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_resized = img_pil.resize((640, 480), Image.BILINEAR)  # back to 640x480 for speed
    img_tensor  = transform(img_resized).unsqueeze(0).cuda()

    with torch.no_grad():
        scores, _, _ = seg_module.decoder(
            seg_module.encoder(img_tensor, return_feature_maps=True))

    probs           = torch.softmax(scores, dim=1)
    max_probs, pred = torch.max(probs, dim=1)
    pred            = pred.squeeze(0).cpu().numpy()
    max_probs       = max_probs.squeeze(0).cpu().numpy()
    pred[max_probs < THRESHOLD] = 0

    pred = cv2.resize(pred.astype(np.uint8), (w, h),
                      interpolation=cv2.INTER_NEAREST)
    return pred


def get_object_boxes(yolo_results, frame_shape, exclude_classes={'person'}):
    """Return list of (x1,y1,x2,y2) for non-person detections."""
    boxes = []
    h, w  = frame_shape[:2]
    for r in yolo_results:
        for box in r.boxes:
            cls_name = yolo.model.names[int(box.cls)]
            if cls_name in exclude_classes:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            boxes.append((x1, y1, x2, y2))
    return boxes


def build_object_mask(boxes, shape):
    """Binary mask — True where any object bounding box exists."""
    mask = np.zeros(shape[:2], dtype=bool)
    for (x1, y1, x2, y2) in boxes:
        mask[y1:y2, x1:x2] = True
    return mask


def get_hand_landmarks(pose_result, frame_shape):
    """Return pixel coords of hand/wrist landmarks (indices 15-22)."""
    HAND_IDS = [15, 16, 17, 18, 19, 20, 21, 22]
    h, w     = frame_shape[:2]
    points   = []
    if pose_result.pose_landmarks:
        for pose_landmarks in pose_result.pose_landmarks:
            all_pts = [(int(lm.x * w), int(lm.y * h))
                       for lm in pose_landmarks]
            for idx in HAND_IDS:
                if idx < len(all_pts):
                    points.append(all_pts[idx])
    return points


def hand_inside_object_box(hand_points, boxes):
    """Return True if any hand landmark is inside any object box."""
    for (px, py) in hand_points:
        for (x1, y1, x2, y2) in boxes:
            if x1 <= px <= x2 and y1 <= py <= y2:
                return True
    return False


def build_hand_region_mask(hand_points, shape, radius=60):
    """Paint a mask covering the hand area using landmark positions."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for (px, py) in hand_points:
        cv2.circle(mask, (px, py), radius, 1, -1)
    return mask.astype(bool)


# ── Main loop ────────────────────────────────────────────────────────────────
cap      = cv2.VideoCapture(0)
frame_ts = 0
prev_time = cv2.getTickCount()

# Cached results from previous frames
last_object_boxes = []
last_object_mask  = None
last_contact_mask = None

print("Running — press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Init cached masks on first frame
    if last_object_mask is None:
        last_object_mask  = np.zeros((h, w), dtype=bool)
        last_contact_mask = np.zeros((h, w), dtype=bool)

    # 1. YOLO — only every YOLO_EVERY frames
    if frame_ts % YOLO_EVERY == 0:
        yolo_results      = yolo(frame, verbose=False, conf=0.15)
        last_object_boxes = get_object_boxes(yolo_results, frame.shape)
        last_object_mask  = build_object_mask(last_object_boxes, frame.shape)

    object_boxes = last_object_boxes
    object_mask  = last_object_mask

    # 2. HOT — only every HOT_EVERY frames
    if frame_ts % HOT_EVERY == 0:
        hot_pred          = run_hot(frame)
        last_contact_mask = (hot_pred > 0)

    contact_mask = last_contact_mask

    # 3. MediaPipe — every frame (lightweight)
    rgb         = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img      = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    pose_result = landmarker.detect_for_video(mp_img, frame_ts)

    # 4. COMBINE — pixel-level touch + grip heuristic
    pixel_contact = contact_mask & object_mask

    hand_pts    = get_hand_landmarks(pose_result, frame.shape)
    hand_region = build_hand_region_mask(hand_pts, frame.shape, radius=60)
    hand_in_box = hand_inside_object_box(hand_pts, object_boxes)

    grip_contact = np.zeros((h, w), dtype=bool)
    if hand_in_box:
        grip_contact = hand_region & object_mask

    valid_contact = pixel_contact | grip_contact

    # 5. Overlay contact mask
    overlay = frame.copy()
    overlay[valid_contact] = (
        overlay[valid_contact] * 0.4 +
        CONTACT_COLOR * 0.6
    ).astype(np.uint8)

    # Draw object boxes (green)
    for (x1, y1, x2, y2) in object_boxes:
        cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,0), 2)

    # 6. Draw skeleton
    if pose_result.pose_landmarks:
        for pose_landmarks in pose_result.pose_landmarks:
            points = []
            for lm in pose_landmarks:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                points.append((x, y))
                cv2.circle(overlay, (x, y), 4, (0,255,0), -1)
            for (s, e) in POSE_CONNECTIONS:
                if s < len(points) and e < len(points):
                    cv2.line(overlay, points[s], points[e], (0,255,255), 2)

    # 7. FPS + HUD
    curr_time = cv2.getTickCount()
    fps       = cv2.getTickFrequency() / (curr_time - prev_time)
    prev_time = curr_time

    contact_type = ("GRIP"  if grip_contact.any()  else
                    "TOUCH" if pixel_contact.any() else "NO")
    cv2.putText(overlay,
        f"Objects: {len(object_boxes)}  Contact: {contact_type}  FPS: {fps:.1f}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("HOT + YOLO Contact Detection", overlay)
    frame_ts += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()