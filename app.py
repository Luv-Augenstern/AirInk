"""
AirInk — Incorporating open-source best practices
References:
  ★ KrShahil/Air-Drawing-System  → multi-finger gesture control for color/thickness/clear
  ★ SakshamShandilya/AirSketch   → adaptive smoothing + distance sampling de-jitter + alpha blending
  ★ IrfanKpm/OpenCv_Paint        → undo stack + pinch detection
  ★ ryanw-2/MediaPipe-Hand-Detector → encapsulated HandDetector class for better reusability

Install dependencies:
    pip install opencv-python mediapipe numpy

Gesture controls:
  ✦ Index finger only (others curled)        → Draw
  ✦ Index + middle fingers                   → Move cursor (no drawing)
  ✦ Index + middle + ring fingers            → Switch to next color
  ✦ Index + pinky (middle/ring bent)         → Increase brush size
  ✦ Index + ring (middle/pinky bent)         → Decrease brush size
  ✦ All five fingers spread (open palm)      → Clear canvas
  ✦ Thumb + index pinch                      → Undo

Keyboard shortcuts:
  c  clear | s  save | q  quit | z  undo | 1-5 colors | +/- thickness
"""

import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque

# Display config
DISPLAY_W, DISPLAY_H = 1280, 720
WINDOW_NAME = "AirInk v2"

# Brush config
COLORS = [
    ("Red",    (0,   0,   255)),
    ("Green",  (0,   200,  80)),
    ("Blue",   (255,  80,   0)),
    ("Yellow", (0,   230, 255)),
    ("White",  (255, 255, 255)),
    ("Purple", (200,  0,  200)),
]
color_idx   = 0
BRUSH_THICK = 6
ERASE_THICK = 30

# Smoothing config
SMOOTH_WINDOW = 5       # fingertip coordinate moving average window
MIN_DRAW_DIST = 3       # distance filter: skip drawing if finger moves less than this (de-jitter)
point_history = deque(maxlen=SMOOTH_WINDOW)

# Undo stack
MAX_UNDO = 20
undo_stack: list[np.ndarray] = []

# Pinch detection
PINCH_DIST_THRESH  = 35   # pixel distance threshold for pinch
PINCH_COOLDOWN     = 0.8  # seconds: prevent repeated triggers
last_pinch_time    = 0.0

# Color switch gesture cooldown
COLOR_GESTURE_COOLDOWN = 1.2
last_color_gesture_time = 0.0

# MediaPipe initialization
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.65,
    model_complexity=0,   # 0=lite, better performance (AirSketch recommendation)
)

# andDetector wrapper
class HandDetector:
    """Wraps MediaPipe hand detection, provides reusable interface"""

    def __init__(self, results):
        self.lm = None
        if results.multi_hand_landmarks:
            self.lm = results.multi_hand_landmarks[0].landmark

    @property
    def detected(self) -> bool:
        return self.lm is not None

    def tip(self, idx: int, w: int, h: int) -> tuple[int, int]:
        """Returns pixel coordinates for the given landmark index"""
        lm = self.lm[idx]
        return int(lm.x * w), int(lm.y * h)

    def fingers_up(self) -> list[bool]:
        """
        Returns [thumb, index, middle, ring, pinky] finger states.
        True = extended, False = curled
        """
        lm = self.lm
        tips  = [4,  8, 12, 16, 20]
        pips  = [3,  6, 10, 14, 18]
        up = []
        # thumb: x-axis check
        up.append(abs(lm[4].x - lm[17].x) > abs(lm[3].x - lm[17].x))
        # other four fingers: y-axis check (tip < pip → extended)
        for tip, pip in zip(tips[1:], pips[1:]):
            up.append(lm[tip].y < lm[pip].y)
        return up

    def finger_count(self) -> int:
        return sum(self.fingers_up())

    def pinch_distance(self, w: int, h: int) -> float:
        """Pixel distance between thumb tip (4) and index tip (8)"""
        t = np.array(self.tip(4, w, h), dtype=float)
        i = np.array(self.tip(8, w, h), dtype=float)
        return float(np.linalg.norm(t - i))

    def landmarks_raw(self):
        return self.lm


# Camera initialization
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

canvas = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, DISPLAY_W, DISPLAY_H)

prev_point  = None
eraser_mode = False

# Helper functions
def smooth_point(new_pt: tuple[int, int]) -> tuple[int, int]:
    """Moving average smoothing to reduce hand shake (from AirSketch)"""
    point_history.append(new_pt)
    xs = [p[0] for p in point_history]
    ys = [p[1] for p in point_history]
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))

def push_undo(canvas: np.ndarray) -> list:
    """Save current canvas snapshot to undo stack"""
    stack = undo_stack.copy()
    stack.append(canvas.copy())
    if len(stack) > MAX_UNDO:
        stack.pop(0)
    return stack

def alpha_blend(frame: np.ndarray, canvas: np.ndarray, alpha=0.85) -> np.ndarray:
    """
    Alpha blend canvas onto camera frame (from AirSketch).
    Stroke opacity controlled by alpha — softer than direct overlay.
    """
    mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 0
    combined = frame.copy()
    combined[mask] = cv2.addWeighted(
        frame, 1 - alpha, canvas, alpha, 0
    )[mask]
    return combined

def draw_ui(img: np.ndarray, hand: HandDetector, status: str, tip_pt):
    """Draw UI elements"""
    h, w = img.shape[:2]
    _, brush_color = COLORS[color_idx]

    # top status bar background
    cv2.rectangle(img, (0, 0), (w, 55), (20, 20, 20), -1)

    # color swatches
    for i, (name, col) in enumerate(COLORS):
        x = 10 + i * 60
        cv2.rectangle(img, (x, 8), (x + 45, 45), col, -1)
        if i == color_idx and not eraser_mode:
            cv2.rectangle(img, (x - 2, 6), (x + 47, 47), (255, 255, 255), 2)

    # eraser indicator
    eraser_x = 10 + len(COLORS) * 60
    label = "Eraser" if eraser_mode else ""
    if eraser_mode:
        cv2.rectangle(img, (eraser_x, 8), (eraser_x + 55, 45), (60, 60, 60), -1)
        cv2.rectangle(img, (eraser_x - 2, 6), (eraser_x + 57, 47), (255, 255, 255), 2)
    cv2.putText(img, label, (eraser_x + 5, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # top-right: status + thickness
    status_color = (80, 255, 120) if "Draw" in status else (100, 100, 255)
    cv2.putText(img, status, (w - 200, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(img, f"Thick:{BRUSH_THICK}", (w - 340, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    # fingertip tracking dot
    if tip_pt:
        color = (128, 128, 128) if eraser_mode else brush_color
        cv2.circle(img, tip_pt, BRUSH_THICK + 4, color, -1)
        cv2.circle(img, tip_pt, BRUSH_THICK + 9, (255, 255, 255), 2)

    # bottom hint bar
    cv2.rectangle(img, (0, h - 28), (w, h), (20, 20, 20), -1)
    hint = "1 finger:draw | 2:move | 3:switch color | index+pinky:thicker | open palm:clear | pinch:undo"
    cv2.putText(img, hint, (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

# Main loop
print("=" * 55)
print("  AirInk v2 — Incorporating open-source best practices")
print("=" * 55)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))

    # MediaPipe detection
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands_detector.process(rgb)
    rgb.flags.writeable = True

    hand = HandDetector(results)
    tip_pt  = None
    drawing = False
    status  = "No hand"

    _, brush_color = COLORS[color_idx]
    now = time.time()

    if hand.detected:
        # Draw hand skeleton
        mp_drawing.draw_landmarks(
            frame,
            results.multi_hand_landmarks[0],
            mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style(),
        )

        fu = hand.fingers_up()          # [thumb, index, middle, ring, pinky]
        n  = sum(fu)
        raw_tip = hand.tip(8, DISPLAY_W, DISPLAY_H)   # index fingertip

        # Gesture recognition
        # index only → draw
        if fu[1] and not fu[2] and not fu[3] and not fu[4]:
            drawing = True
            status  = "✏ Drawing"
            tip_pt  = smooth_point(raw_tip)   # smoothing

        # index + middle → move (no drawing)
        elif fu[1] and fu[2] and not fu[3] and not fu[4]:
            status  = "✋ Moving"
            tip_pt  = smooth_point(raw_tip)
            prev_point = None

        # index + middle + ring → switch color
        elif fu[1] and fu[2] and fu[3] and not fu[4]:
            status = "🎨 Switch color"
            if now - last_color_gesture_time > COLOR_GESTURE_COOLDOWN:
                color_idx = (color_idx + 1) % len(COLORS)
                eraser_mode = False
                last_color_gesture_time = now
                print(f"[Color] → {COLORS[color_idx][0]}")
            prev_point = None

        # index + pinky (others bent) → increase brush
        elif fu[1] and not fu[2] and not fu[3] and fu[4]:
            status = "➕ Increase brush"
            if now - last_color_gesture_time > 0.4:
                BRUSH_THICK = min(BRUSH_THICK + 2, 40)
                last_color_gesture_time = now
            prev_point = None

        # index + ring (others bent) → decrease brush
        elif fu[1] and not fu[2] and fu[3] and not fu[4]:
            status = "➖ Decrease brush"
            if now - last_color_gesture_time > 0.4:
                BRUSH_THICK = max(BRUSH_THICK - 2, 1)
                last_color_gesture_time = now
            prev_point = None

        # all five fingers open → clear canvas
        elif n >= 5:
            status = "🖐 Clear"
            if now - last_color_gesture_time > 1.0:
                undo_stack.extend([canvas.copy()])  # push snapshot before clearing
                canvas[:] = 0
                prev_point = None
                last_color_gesture_time = now
                print("[Clear] Canvas reset")

        else:
            status = f"({n} fingers)"
            prev_point = None

        # Pinch detection = undo
        pinch = hand.pinch_distance(DISPLAY_W, DISPLAY_H)
        if pinch < PINCH_DIST_THRESH and now - last_pinch_time > PINCH_COOLDOWN:
            if undo_stack:
                canvas = undo_stack.pop()
                print("[Undo] Undo")
            last_pinch_time = now
            prev_point = None

    # Drawing logic
    if drawing and tip_pt:
        if prev_point:
            dist = np.linalg.norm(np.array(tip_pt) - np.array(prev_point))
            if dist > MIN_DRAW_DIST:   # distance filter de-jitter
                # save undo snapshot (before each stroke)
                if dist > 30 and len(undo_stack) < MAX_UNDO:
                    undo_stack = push_undo(canvas)
                if eraser_mode:
                    cv2.line(canvas, prev_point, tip_pt, (0, 0, 0), ERASE_THICK)
                else:
                    cv2.line(canvas, prev_point, tip_pt, brush_color, BRUSH_THICK)
        prev_point = tip_pt
    else:
        if not drawing:
            prev_point = None

    # Alpha blending 
    combined = alpha_blend(frame, canvas, alpha=0.92)

    # UI drawing
    draw_ui(combined, hand, status, tip_pt)

    cv2.imshow(WINDOW_NAME, combined)

    # Keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        undo_stack = push_undo(canvas)
        canvas[:] = 0
        prev_point = None
    elif key == ord('z'):
        if undo_stack:
            canvas = undo_stack.pop()
            print("[Undo]")
    elif key == ord('s'):
        cv2.imwrite("drawing.png", canvas)
        print("[Save] drawing.png")
    elif key == ord('+') or key == ord('='):
        BRUSH_THICK = min(BRUSH_THICK + 2, 40)
    elif key == ord('-'):
        BRUSH_THICK = max(BRUSH_THICK - 2, 1)
    elif key == ord('e'):
        eraser_mode = not eraser_mode
    elif ord('1') <= key <= ord('6'):
        idx = key - ord('1')
        if idx < len(COLORS):
            color_idx = idx
            eraser_mode = False

# Cleanup
hands_detector.close()
cap.release()
cv2.destroyAllWindows()
