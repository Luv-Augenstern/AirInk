import cv2
import numpy as np

# Tracked color: "blue" objects (e.g., blue bottle cap)
# HSV range can be fine-tuned (blue Hue typically 100~130)
LOWER_COLOR = np.array([100, 80, 80])   # HSV lower bound
UPPER_COLOR = np.array([130, 255, 255]) # HSV upper bound

BRUSH_COLOR  = (0, 0, 255)   # Brush color (BGR), default red
BRUSH_THICK  = 5             # Brush thickness
MIN_CONTOUR_AREA = 500       # Minimum contour area to filter noise

# Initialization
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera. Please check device connection.")

ret, frame = cap.read()
if not ret:
    raise RuntimeError("Cannot read frame from camera.")

h, w = frame.shape[:2]
canvas = np.zeros((h, w, 3), dtype=np.uint8)  # Persistent drawing canvas

prev_point = None   # Previous frame's tip coordinate

print("=" * 45)
print("  AirInk - Basic Tracking & Drawing")
print("  Tracking color: Blue object (e.g., blue bottle cap)")
print("  Hotkeys:")
print("    c  — Clear canvas")
print("    s  — Save as drawing.png")
print("    q  — Quit")
print("=" * 45)

# Color → Brush Mapping (press number keys to switch color)
COLOR_MAP = {
    ord('1'): ("Red",   (0,   0,   255)),
    ord('2'): ("Green", (0,   255,  0 )),
    ord('3'): ("Blue",  (255,  0,   0 )),
    ord('4'): ("Yellow",(0,   255, 255)),
    ord('5'): ("White", (255, 255, 255)),
}

# Main Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally for mirror effect
    frame = cv2.flip(frame, 1)

    # ── Color Tracking ──────────────────────
    hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask  = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)

    # Morphological denoising: erode then dilate
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode (mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find the largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    tip = None  # Current frame's tip position
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > MIN_CONTOUR_AREA:
            # Compute contour centroid
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                tip = (cx, cy)

                # Draw tracking circle on the original frame
                cv2.circle(frame, tip, 8, BRUSH_COLOR, -1)
                cv2.circle(frame, tip, 12, (255, 255, 255), 2)

    # Draw onto Canvas
    if tip and prev_point:
        cv2.line(canvas, prev_point, tip, BRUSH_COLOR, BRUSH_THICK)

    prev_point = tip  # Break line when tip leaves tracking area

    # Combine Canvas with Live Frame
    # Overlay canvas onto camera feed (non-zero pixels override)
    combined = frame.copy()
    overlay_mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 0
    combined[overlay_mask] = canvas[overlay_mask]

    # On-Screen Text Hints
    cv2.putText(combined, f"Color: {BRUSH_COLOR}  Thick: {BRUSH_THICK}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(combined, "c:clear  s:save  q:quit  1-5:color",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.imshow("AirInk", combined)

    # Keyboard Responses
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):                          # Quit
        break
    elif key == ord('c'):                        # Clear canvas
        canvas[:] = 0
        prev_point = None
        print("[Clear] Canvas reset")
    elif key == ord('s'):                        # Save drawing
        cv2.imwrite("drawing.png", canvas)
        print("[Save] drawing.png")
    elif key == ord('+') or key == ord('='):     # Increase thickness
        BRUSH_THICK = min(BRUSH_THICK + 2, 40)
    elif key == ord('-'):                        # Decrease thickness
        BRUSH_THICK = max(BRUSH_THICK - 2, 1)
    elif key in COLOR_MAP:                       # Switch color
        name, BRUSH_COLOR = COLOR_MAP[key]
        print(f"[Color] Switched to {name}")

# ─────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()