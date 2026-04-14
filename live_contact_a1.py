import cv2
import torch
import numpy as np
import mediapipe as mp
import sys, os
import csv
from datetime import datetime
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from collections import deque

# ── Paths ────────────────────────────────────────────────────────────────────
HOT_ROOT = os.path.join(os.path.dirname(__file__), 'HOT', 'HOT-main')
sys.path.insert(0, HOT_ROOT)

from hot.models import ModelBuilder, SegmentationModule
from hot.config import cfg

CFG_FILE  = os.path.join(HOT_ROOT, 'config', 'hot-resnet50dilated-c1.yaml')
CKPT_DIR  = os.path.join(HOT_ROOT, 'ckpt', 'hot-c1')

# ── Per-class thresholds (TUNING 4) ──────────────────────────────────────────
# HOT classes: 0=background, 1-10=body contact regions, 11-17=object regions
# Lower threshold for hand classes (1,2) = more sensitive
# Higher for background = stricter
PER_CLASS_THRESHOLD = {
    0:  0.65,   # background — strict
    1:  0.45,   # left hand
    2:  0.45,   # right hand
    3:  0.50,   # left arm
    4:  0.50,   # right arm
    5:  0.55,   # torso
    6:  0.55,   # left leg
    7:  0.55,   # right leg
    8:  0.50,   # left foot
    9:  0.50,   # right foot
    10: 0.55,   # head
}
DEFAULT_THRESHOLD = 0.50  # fallback for classes not listed

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

# ── Load YOLO — upgraded to yolov8s (TUNING 1) ───────────────────────────────
yolo = YOLO("yolov8s.pt")  # auto-downloads ~22MB

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

# ── Stability + tuning settings ───────────────────────────────────────────────
YOLO_EVERY     = 10
HOT_EVERY      = 2   # TUNING 2: was 3, now 2 for smoother mask
SMOOTH_BUFFER  = 7   # TUNING 5: was 5, now 7 for more stability
PERSIST_FRAMES = 15

# ── Diagnostics settings ──────────────────────────────────────────────────────
LOG_ENABLED = True
os.makedirs("logs", exist_ok=True)
LOG_FILE    = f"logs/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# ── Helper functions ──────────────────────────────────────────────────────────
def apply_per_class_threshold(scores, pred, max_probs):
    """Apply different thresholds per HOT class (TUNING 4)."""
    result = pred.copy()
    for cls_id, threshold in PER_CLASS_THRESHOLD.items():
        cls_mask = (pred == cls_id)
        result[cls_mask & (max_probs < threshold)] = 0
    # Apply default threshold to any class not in dict
    listed = set(PER_CLASS_THRESHOLD.keys())
    for cls_id in np.unique(pred):
        if cls_id not in listed:
            cls_mask = (pred == cls_id)
            result[cls_mask & (max_probs < DEFAULT_THRESHOLD)] = 0
    return result


def run_hot(frame):
    h, w        = frame.shape[:2]
    img_pil     = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_resized = img_pil.resize((640, 480), Image.BILINEAR)
    img_tensor  = transform(img_resized).unsqueeze(0).cuda()

    with torch.no_grad():
        scores, _, _ = seg_module.decoder(
            seg_module.encoder(img_tensor, return_feature_maps=True))

    probs           = torch.softmax(scores, dim=1)
    max_probs, pred = torch.max(probs, dim=1)
    pred            = pred.squeeze(0).cpu().numpy()
    max_probs       = max_probs.squeeze(0).cpu().numpy()

    # Per-class threshold instead of single global threshold
    pred = apply_per_class_threshold(scores, pred, max_probs)

    pred = cv2.resize(pred.astype(np.uint8), (w, h),
                      interpolation=cv2.INTER_NEAREST)
    return pred


def get_object_boxes(yolo_results, frame_shape, exclude_classes={'person'}):
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
    mask = np.zeros(shape[:2], dtype=bool)
    for (x1, y1, x2, y2) in boxes:
        mask[y1:y2, x1:x2] = True
    return mask


def get_hand_landmarks(pose_result, frame_shape):
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
    for (px, py) in hand_points:
        for (x1, y1, x2, y2) in boxes:
            if x1 <= px <= x2 and y1 <= py <= y2:
                return True
    return False


def get_dynamic_radius(pose_result, frame_shape):
    """
    TUNING 3: Dynamic radius based on shoulder width.
    Wider shoulders in frame = closer to camera = larger radius.
    """
    LEFT_SHOULDER  = 11
    RIGHT_SHOULDER = 12
    default_radius = 60
    h, w           = frame_shape[:2]

    if not pose_result.pose_landmarks:
        return default_radius

    for pose_landmarks in pose_result.pose_landmarks:
        all_pts = [(lm.x * w, lm.y * h) for lm in pose_landmarks]
        if LEFT_SHOULDER < len(all_pts) and RIGHT_SHOULDER < len(all_pts):
            lx, ly = all_pts[LEFT_SHOULDER]
            rx, ry = all_pts[RIGHT_SHOULDER]
            shoulder_width = abs(rx - lx)
            # Scale radius: shoulder ~200px = radius 60, ~400px = radius 100
            radius = int(np.clip(shoulder_width * 0.3, 35, 120))
            return radius

    return default_radius


def build_hand_region_mask(hand_points, shape, radius=60):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for (px, py) in hand_points:
        cv2.circle(mask, (px, py), radius, 1, -1)
    return mask.astype(bool)


# ── Main loop ─────────────────────────────────────────────────────────────────
cap       = cv2.VideoCapture(0)
frame_ts  = 0
prev_time = cv2.getTickCount()

# Cached results
last_object_boxes = []
last_object_mask  = None
last_contact_mask = None

# Smoothing buffer
hot_mask_buffer = deque(maxlen=SMOOTH_BUFFER)

# Persistence state
frames_since_contact = PERSIST_FRAMES + 1
persisted_mask       = None

# Diagnostics state
log_rows              = []
contact_state_prev    = False
contact_flicker_count = 0
false_positive_frames = 0
total_frames          = 0
contact_frames        = 0
session_start         = datetime.now()

print("Running — press Q to quit")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # Init cached masks on first frame
        if last_object_mask is None:
            last_object_mask  = np.zeros((h, w), dtype=bool)
            last_contact_mask = np.zeros((h, w), dtype=bool)
            persisted_mask    = np.zeros((h, w), dtype=bool)

        # 1. YOLO — every YOLO_EVERY frames
        if frame_ts % YOLO_EVERY == 0:
            yolo_results      = yolo(frame, verbose=False, conf=0.25)
            last_object_boxes = get_object_boxes(yolo_results, frame.shape)
            last_object_mask  = build_object_mask(last_object_boxes, frame.shape)

        object_boxes = last_object_boxes
        object_mask  = last_object_mask

        # 2. HOT — every HOT_EVERY frames
        if frame_ts % HOT_EVERY == 0:
            hot_pred = run_hot(frame)
            raw_mask = (hot_pred > 0)

            hot_mask_buffer.append(raw_mask.astype(np.uint8))
            if len(hot_mask_buffer) >= 3:
                stacked       = np.stack(list(hot_mask_buffer), axis=0)
                smoothed_mask = (stacked.sum(axis=0) >= (len(hot_mask_buffer) // 2 + 1))
            else:
                smoothed_mask = raw_mask

            last_contact_mask = smoothed_mask

        contact_mask = last_contact_mask

        # 3. MediaPipe — every frame
        rgb         = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img      = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        pose_result = landmarker.detect_for_video(mp_img, frame_ts)

        # 4. COMBINE — pixel touch + grip heuristic
        pixel_contact = contact_mask & object_mask

        hand_pts      = get_hand_landmarks(pose_result, frame.shape)
        dynamic_radius = get_dynamic_radius(pose_result, frame.shape)
        hand_region   = build_hand_region_mask(hand_pts, frame.shape,
                                               radius=dynamic_radius)
        hand_in_box   = hand_inside_object_box(hand_pts, object_boxes)

        grip_contact = np.zeros((h, w), dtype=bool)
        if hand_in_box:
            grip_contact = hand_region & object_mask

        current_contact = pixel_contact | grip_contact

        # 5. Persistence
        if current_contact.any():
            frames_since_contact = 0
            persisted_mask       = current_contact.copy()
        else:
            frames_since_contact += 1

        if frames_since_contact <= PERSIST_FRAMES:
            valid_contact = persisted_mask
            is_persisting = frames_since_contact > 0
        else:
            valid_contact = current_contact
            is_persisting = False

        # 6. Overlay — fade out persisted contact
        overlay = frame.copy()
        if valid_contact.any():
            alpha = 0.6
            if is_persisting:
                alpha = 0.6 * (1 - frames_since_contact / PERSIST_FRAMES)
            overlay[valid_contact] = (
                overlay[valid_contact] * (1 - alpha) +
                CONTACT_COLOR * alpha
            ).astype(np.uint8)

        # Draw object boxes with class label
        for r in (yolo_results if frame_ts % YOLO_EVERY == 0
                  else [type('R', (), {'boxes': []})()]):
            pass
        for (x1, y1, x2, y2) in object_boxes:
            cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,0), 2)

        # 7. Skeleton
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

        # 8. FPS + HUD
        curr_time = cv2.getTickCount()
        fps       = cv2.getTickFrequency() / (curr_time - prev_time)
        prev_time = curr_time

        contact_type = ("PERSIST" if is_persisting      else
                        "GRIP"    if grip_contact.any()  else
                        "TOUCH"   if pixel_contact.any() else "NO")

        cv2.putText(overlay,
            f"Objects: {len(object_boxes)}  Contact: {contact_type}  FPS: {fps:.1f}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # Show dynamic radius on HUD for tuning visibility
        cv2.putText(overlay,
            f"Radius: {dynamic_radius}px  Threshold: per-class  YOLO: s",
            (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        # 9. Diagnostics
        total_frames   += 1
        contact_active  = valid_contact.any()

        if contact_active:
            contact_frames += 1

        if contact_active != contact_state_prev:
            contact_flicker_count += 1
        contact_state_prev = contact_active

        if contact_active and len(object_boxes) == 0:
            false_positive_frames += 1

        if LOG_ENABLED:
            log_rows.append({
                "frame":           frame_ts,
                "timestamp":       datetime.now().strftime('%H:%M:%S.%f')[:-3],
                "fps":             round(fps, 1),
                "objects":         len(object_boxes),
                "contact_type":    contact_type,
                "contact_active":  int(contact_active),
                "is_persisting":   int(is_persisting),
                "flickers_so_far": contact_flicker_count,
                "dynamic_radius":  dynamic_radius,
            })

        # Write log every 300 frames
        if LOG_ENABLED and frame_ts % 300 == 0 and frame_ts > 0:
            with open(LOG_FILE, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
                writer.writeheader()
                writer.writerows(log_rows)

        cv2.imshow("HOT + YOLO Contact Detection", overlay)
        frame_ts += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nInterrupted — saving log...")

finally:
    cap.release()
    cv2.destroyAllWindows()

    # Write final log
    if LOG_ENABLED and log_rows:
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"Session log saved to {LOG_FILE}")

    # Session summary
    duration = (datetime.now() - session_start).seconds
    print("\n═══════════════════════════════════════")
    print("         SESSION DIAGNOSTICS REPORT    ")
    print("═══════════════════════════════════════")
    print(f"  Duration:            {duration}s")
    print(f"  Total frames:        {total_frames}")
    print(f"  Contact frames:      {contact_frames} ({100*contact_frames//max(total_frames,1)}%)")
    print(f"  Contact flickers:    {contact_flicker_count}")
    print(f"  Flicker rate:        {round(contact_flicker_count/max(duration,1), 2)}/sec")
    print(f"  False pos frames:    {false_positive_frames}")
    print(f"  Log saved to:        {LOG_FILE}")
    print("═══════════════════════════════════════\n")