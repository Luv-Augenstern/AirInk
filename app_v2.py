"""
AirInk v2 — 融合开源项目最佳实践
=====================================================
借鉴来源：
  ★ KrShahil/Air-Drawing-System  → 多指手势控制颜色/粗细/清屏
  ★ SakshamShandilya/AirSketch   → 自适应平滑 + 距离采样去抖动 + Alpha 融合
  ★ IrfanKpm/OpenCv_Paint        → 撤销栈(Undo) + 捏合点击检测
  ★ ryanw-2/MediaPipe-Hand-Detector → 封装 HandDetector 类，复用性更好

安装依赖：
    pip install opencv-python mediapipe numpy

手势操作：
  ✦ 仅食指伸出（其余握拳）       → 绘画
  ✦ 食指 + 中指伸出              → 移动光标（不绘画）
  ✦ 食指 + 中指 + 无名指伸出     → 切换到下一个颜色
  ✦ 食指 + 小指伸出（中指/无名指弯）→ 增大笔刷
  ✦ 食指 + 无名指伸出（中指/小指弯）→ 减小笔刷
  ✦ 五指全展（手掌展开）         → 清空画布
  ✦ 大拇指 + 食指捏合            → 撤销（Undo）

键盘快捷键：
  c  清空 | s  保存 | q  退出 | z  撤销 | 1-5 颜色 | +/- 粗细
"""

import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque

# ─── 显示配置 ─────────────────────────────────────────
DISPLAY_W, DISPLAY_H = 1280, 720
WINDOW_NAME = "AirInk v2"

# ─── 笔刷配置 ─────────────────────────────────────────
COLORS = [
    ("红", (0,   0,   255)),
    ("绿", (0,   200,  80)),
    ("蓝", (255,  80,   0)),
    ("黄", (0,   230, 255)),
    ("白", (255, 255, 255)),
    ("紫", (200,  0,  200)),
]
color_idx   = 0
BRUSH_THICK = 6
ERASE_THICK = 30

# ─── 平滑配置（来自 AirSketch）────────────────────────
SMOOTH_WINDOW = 5       # 指尖坐标滑动平均窗口
MIN_DRAW_DIST = 3       # 距离过滤：手指移动小于此像素不画线（去微抖）
point_history = deque(maxlen=SMOOTH_WINDOW)

# ─── 撤销栈（来自 IrfanKpm/OpenCv_Paint）──────────────
MAX_UNDO = 20
undo_stack: list[np.ndarray] = []

# ─── 捏合检测（来自 IrfanKpm）──────────────────────────
PINCH_DIST_THRESH  = 35   # 像素距离：认为发生捏合
PINCH_COOLDOWN     = 0.8  # 秒：防止连续触发
last_pinch_time    = 0.0

# ─── 颜色切换手势冷却 ─────────────────────────────────
COLOR_GESTURE_COOLDOWN = 1.2
last_color_gesture_time = 0.0

# ─── MediaPipe 初始化 ─────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.65,
    model_complexity=0,   # 0=轻量，性能更好（来自 AirSketch 建议）
)

# ─── HandDetector 封装（参考 ryanw-2 的设计思路）────────
class HandDetector:
    """封装 MediaPipe 手部检测，提供复用接口"""

    def __init__(self, results):
        self.lm = None
        if results.multi_hand_landmarks:
            self.lm = results.multi_hand_landmarks[0].landmark

    @property
    def detected(self) -> bool:
        return self.lm is not None

    def tip(self, idx: int, w: int, h: int) -> tuple[int, int]:
        """返回指定 landmark 的像素坐标"""
        lm = self.lm[idx]
        return int(lm.x * w), int(lm.y * h)

    def fingers_up(self) -> list[bool]:
        """
        返回 [拇指, 食指, 中指, 无名指, 小指] 是否伸直
        True = 伸直，False = 弯曲
        """
        lm = self.lm
        tips  = [4,  8, 12, 16, 20]
        pips  = [3,  6, 10, 14, 18]
        up = []
        # 拇指：x 轴判断
        up.append(abs(lm[4].x - lm[17].x) > abs(lm[3].x - lm[17].x))
        # 其余四指：y 轴判断（指尖 < 中间关节 → 伸直）
        for tip, pip in zip(tips[1:], pips[1:]):
            up.append(lm[tip].y < lm[pip].y)
        return up

    def finger_count(self) -> int:
        return sum(self.fingers_up())

    def pinch_distance(self, w: int, h: int) -> float:
        """拇指尖（4）与食指尖（8）的像素距离"""
        t = np.array(self.tip(4, w, h), dtype=float)
        i = np.array(self.tip(8, w, h), dtype=float)
        return float(np.linalg.norm(t - i))

    def landmarks_raw(self):
        return self.lm


# ─── 摄像头初始化 ──────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

canvas = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, DISPLAY_W, DISPLAY_H)

prev_point  = None
eraser_mode = False

# ─── 辅助函数 ──────────────────────────────────────────
def smooth_point(new_pt: tuple[int, int]) -> tuple[int, int]:
    """滑动平均平滑，减少手抖（来自 AirSketch）"""
    point_history.append(new_pt)
    xs = [p[0] for p in point_history]
    ys = [p[1] for p in point_history]
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))

def push_undo(canvas: np.ndarray) -> list:
    """保存当前画布快照到撤销栈"""
    stack = undo_stack.copy()
    stack.append(canvas.copy())
    if len(stack) > MAX_UNDO:
        stack.pop(0)
    return stack

def alpha_blend(frame: np.ndarray, canvas: np.ndarray, alpha=0.85) -> np.ndarray:
    """
    Alpha 混合：画布叠加到摄像头画面（来自 AirSketch）
    笔迹不透明度由 alpha 控制，比直接覆盖更柔和
    """
    mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 0
    combined = frame.copy()
    combined[mask] = cv2.addWeighted(
        frame, 1 - alpha, canvas, alpha, 0
    )[mask]
    return combined

def draw_ui(img: np.ndarray, hand: HandDetector, status: str, tip_pt):
    """绘制 UI 元素"""
    h, w = img.shape[:2]
    _, brush_color = COLORS[color_idx]

    # 顶部状态栏背景
    cv2.rectangle(img, (0, 0), (w, 55), (20, 20, 20), -1)

    # 颜色色块
    for i, (name, col) in enumerate(COLORS):
        x = 10 + i * 60
        cv2.rectangle(img, (x, 8), (x + 45, 45), col, -1)
        if i == color_idx and not eraser_mode:
            cv2.rectangle(img, (x - 2, 6), (x + 47, 47), (255, 255, 255), 2)

    # 橡皮擦标识
    eraser_x = 10 + len(COLORS) * 60
    label = "橡皮" if eraser_mode else ""
    if eraser_mode:
        cv2.rectangle(img, (eraser_x, 8), (eraser_x + 55, 45), (60, 60, 60), -1)
        cv2.rectangle(img, (eraser_x - 2, 6), (eraser_x + 57, 47), (255, 255, 255), 2)
    cv2.putText(img, label, (eraser_x + 5, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # 右上：状态 + 粗细
    status_color = (80, 255, 120) if "绘画" in status else (100, 100, 255)
    cv2.putText(img, status, (w - 200, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(img, f"粗细:{BRUSH_THICK}", (w - 340, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    # 指尖追踪点
    if tip_pt:
        color = (128, 128, 128) if eraser_mode else brush_color
        cv2.circle(img, tip_pt, BRUSH_THICK + 4, color, -1)
        cv2.circle(img, tip_pt, BRUSH_THICK + 9, (255, 255, 255), 2)

    # 底部提示
    cv2.rectangle(img, (0, h - 28), (w, h), (20, 20, 20), -1)
    hint = "手势:1指绘画 | 2指移动 | 3指换色 | 食指+小指加粗 | 展开手掌清屏 | 捏合撤销"
    cv2.putText(img, hint, (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

# ─── 主循环 ────────────────────────────────────────────
print("=" * 55)
print("  AirInk v2  — 融合开源最佳实践")
print("=" * 55)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))

    # MediaPipe 检测
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands_detector.process(rgb)
    rgb.flags.writeable = True

    hand = HandDetector(results)
    tip_pt  = None
    drawing = False
    status  = "无手"

    _, brush_color = COLORS[color_idx]
    now = time.time()

    if hand.detected:
        # 绘制骨架
        mp_drawing.draw_landmarks(
            frame,
            results.multi_hand_landmarks[0],
            mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style(),
        )

        fu = hand.fingers_up()          # [拇, 食, 中, 无名, 小]
        n  = sum(fu)
        raw_tip = hand.tip(8, DISPLAY_W, DISPLAY_H)   # 食指指尖

        # ── 手势识别（来自 KrShahil/Air-Drawing-System）──
        # 仅食指伸出 → 绘画
        if fu[1] and not fu[2] and not fu[3] and not fu[4]:
            drawing = True
            status  = "✏ 绘画中"
            tip_pt  = smooth_point(raw_tip)   # 平滑

        # 食指 + 中指 → 移动（不绘画）
        elif fu[1] and fu[2] and not fu[3] and not fu[4]:
            status  = "✋ 移动"
            tip_pt  = smooth_point(raw_tip)
            prev_point = None

        # 食指 + 中指 + 无名指 → 切换颜色
        elif fu[1] and fu[2] and fu[3] and not fu[4]:
            status = "🎨 换色手势"
            if now - last_color_gesture_time > COLOR_GESTURE_COOLDOWN:
                color_idx = (color_idx + 1) % len(COLORS)
                eraser_mode = False
                last_color_gesture_time = now
                print(f"[换色] → {COLORS[color_idx][0]}")
            prev_point = None

        # 食指 + 小指（其余弯） → 增大笔刷
        elif fu[1] and not fu[2] and not fu[3] and fu[4]:
            status = "➕ 增大笔刷"
            if now - last_color_gesture_time > 0.4:
                BRUSH_THICK = min(BRUSH_THICK + 2, 40)
                last_color_gesture_time = now
            prev_point = None

        # 食指 + 无名指（其余弯） → 减小笔刷
        elif fu[1] and not fu[2] and fu[3] and not fu[4]:
            status = "➖ 减小笔刷"
            if now - last_color_gesture_time > 0.4:
                BRUSH_THICK = max(BRUSH_THICK - 2, 1)
                last_color_gesture_time = now
            prev_point = None

        # 五指展开 → 清屏
        elif n >= 5:
            status = "🖐 清空"
            if now - last_color_gesture_time > 1.0:
                undo_stack.extend([canvas.copy()])  # 清屏前先入栈
                canvas[:] = 0
                prev_point = None
                last_color_gesture_time = now
                print("[清空] 画布已重置")

        else:
            status = f"({n}指)"
            prev_point = None

        # ── 捏合检测 = 撤销（来自 IrfanKpm）───────────────
        pinch = hand.pinch_distance(DISPLAY_W, DISPLAY_H)
        if pinch < PINCH_DIST_THRESH and now - last_pinch_time > PINCH_COOLDOWN:
            if undo_stack:
                canvas = undo_stack.pop()
                print("[撤销] Undo")
            last_pinch_time = now
            prev_point = None

    # ── 绘画逻辑 ──────────────────────────────────────
    if drawing and tip_pt:
        if prev_point:
            dist = np.linalg.norm(np.array(tip_pt) - np.array(prev_point))
            if dist > MIN_DRAW_DIST:   # 距离过滤去抖（AirSketch 技巧）
                # 保存 undo 快照（每次起笔前）
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

    # ── Alpha 混合（AirSketch 技术）──────────────────
    combined = alpha_blend(frame, canvas, alpha=0.92)

    # ── UI 绘制 ────────────────────────────────────
    draw_ui(combined, hand, status, tip_pt)

    cv2.imshow(WINDOW_NAME, combined)

    # ── 键盘响应 ───────────────────────────────────
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
            print("[撤销]")
    elif key == ord('s'):
        cv2.imwrite("drawing.png", canvas)
        print("[保存] drawing.png")
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

# ─── 清理 ──────────────────────────────────────────────
hands_detector.close()
cap.release()
cv2.destroyAllWindows()
