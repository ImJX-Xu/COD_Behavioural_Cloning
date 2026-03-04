# 该文件基于 https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning 的 screen_input / 采集流程思路改写
# 原作者：Tim Pearce, Jun Zhu
# 衍生项目：COD_BC (https://github.com/ImJX-Xu/COD_Behavioural_Cloning)，适配 COD 与 mss/pynput
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import h5py
import mss
import numpy as np
from pynput import keyboard, mouse

try:
    import win32gui  # type: ignore[import-untyped]
    import win32con  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - optional on non-Windows
    win32gui = None  # type: ignore[assignment]
    win32con = None  # type: ignore[assignment]


GAME_WINDOW_TITLE_KEYWORDS = ["Call of Duty", "COD", "使命召唤"]

# 采样帧率：与 Example 一致（config.TARGET_FPS）
from config import TARGET_FPS
FRAME_INTERVAL = 1.0 / TARGET_FPS

# 训练所需分辨率由 config.USE_EFFICIENTNET 决定，见下方 TARGET_SIZE 赋值

# 主画面：以屏幕中心为中心、大小为整屏 1/2（宽高各一半）
#
# 小地图：基于圆心定位，(x_center, y_center, r) 相对全屏比例 0~1
#   - x_center: 圆心相对于屏幕宽的比例（0=最左，1=最右）
#   - y_center: 圆心相对于屏幕高的比例（0=最上，1=最下）
#   - r: 半径，相对于屏幕高的比例；正方形框边长 = 2r（高方向），宽方向取 2r*H/W 使裁切为像素正方形
#   - 正方形框：中心 (x_center, y_center)，边长 2r（高方向），框内再做圆遮罩
MINIMAP_CENTER_REL = (0.105, 0.168, 0.12)  # (x_center, y_center, r)

# 录制控制按键
START_STOP_KEY = keyboard.Key.f9  # 开始/暂停录制
EXIT_KEY = keyboard.Key.f12       # 结束进程并保存

# 功能开关与尺寸：从总控 config 读取
from config import (
    USE_EFFICIENTNET,
    IMG_HEIGHT,
    IMG_WIDTH,
    USE_MINIMAP,
    MAIN_TARGET_SIZE,
    MINIMAP_TARGET_SIZE,
)


@dataclass
class ActionRecord:
    timestamp: float
    mouse_dx: float
    mouse_dy: float
    mouse_left: int
    mouse_right: int
    keys: List[str]


class InputState:
    def __init__(self):
        self.pressed_keys = set()
        self.last_mouse_pos = None
        self.mouse_left = 0
        self.mouse_right = 0

    def get_keys_snapshot(self) -> List[str]:
        return list(self.pressed_keys)


def get_fullscreen_monitor(sct: mss.mss) -> Dict[str, int]:
    """主显示器全屏区域（用于 grab，再按需裁主画面为下方 2/3）。"""
    mon = sct.monitors[1]
    return {
        "left": mon["left"],
        "top": mon["top"],
        "width": mon["width"],
        "height": mon["height"],
    }


def crop_main_center_half(img: np.ndarray, monitor: Dict[str, int]) -> np.ndarray:
    """主画面：以屏幕中心为中心、大小为整屏 1/2（宽高各取一半）。img 全屏 (H, W, C)，返回 (H/2, W/2, C)。"""
    h, w = monitor["height"], monitor["width"]
    y1, y2 = h // 4, h * 3 // 4
    x1, x2 = w // 4, w * 3 // 4
    return img[y1:y2, x1:x2]


def preprocess_frame(img_bgra: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """target_size: (width, height) for cv2.resize."""
    img_rgb = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGB)
    resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8)  # (height, width, 3)


def _minimap_center_to_roi_abs(
    monitor: Dict[str, int],
    center_rel: Tuple[float, float, float],
) -> Tuple[int, int, int, int]:
    """
    由圆心 (x_center, y_center, r) 得到小地图正方形框的像素 (x, y, w, h)。
    x_center, y_center, r 为相对比例 0~1；r 为半径（相对屏幕高），正方形边长 = 2r（高方向），
    宽方向按 2r*H/W 取，使裁切区域为像素正方形。
    """
    x_center, y_center, r = center_rel
    W, H = monitor["width"], monitor["height"]
    # 正方形边长（像素）= 2*r*H；圆心像素
    half_side_px = int(r * H)
    side_px = max(2, 2 * half_side_px)
    cx_px = int(x_center * W)
    cy_px = int(y_center * H)
    abs_x = cx_px - side_px // 2
    abs_y = cy_px - side_px // 2
    abs_x = max(0, min(abs_x, W - side_px))
    abs_y = max(0, min(abs_y, H - side_px))
    abs_w = min(side_px, W - abs_x)
    abs_h = min(side_px, H - abs_y)
    return abs_x, abs_y, abs_w, abs_h


def preprocess_frame_roi(
    img_bgra: np.ndarray,
    roi_xywh: Tuple[int, int, int, int],
    target_size: Tuple[int, int] = None,
) -> np.ndarray:
    x, y, w, h = roi_xywh
    cropped = img_bgra[y : y + h, x : x + w]
    if target_size is None:
        target_size = MINIMAP_TARGET_SIZE
    return preprocess_frame(cropped, target_size)


def apply_circular_minimap_mask(img: np.ndarray) -> np.ndarray:
    """
    对小地图裁切图施加圆形遮罩：圆内保留，圆外置黑。
    圆心取裁切图的几何中心 (w/2, h/2)，半径 = min(w,h)/2（内接圆）。
    圆心为裁切图几何中心；屏幕上的圆心由 MINIMAP_CENTER_REL 的 (x_center, y_center, r) 决定。
    img 为 (H, W) 或 (H, W, 3)，原地修改并返回。
    """
    h, w = img.shape[0], img.shape[1]
    cx, cy = w // 2, h // 2   # 圆心：裁切图中心（调圆心/半径请改 MINIMAP_CENTER_REL）
    r = min(w, h) // 2
    y_grid, x_grid = np.ogrid[:h, :w]
    mask = ((x_grid - cx) ** 2 + (y_grid - cy) ** 2 <= r**2).astype(img.dtype)
    if img.ndim == 3:
        mask = mask[:, :, np.newaxis]
    img[:] = img * mask
    return img


def encode_action(
    prev_mouse_pos: Tuple[int, int],
    curr_mouse_pos: Tuple[int, int],
    input_state: InputState,
    timestamp: float,
) -> ActionRecord:
    if prev_mouse_pos is None:
        mouse_dx, mouse_dy = 0.0, 0.0
    else:
        mouse_dx = float(curr_mouse_pos[0] - prev_mouse_pos[0])
        mouse_dy = float(curr_mouse_pos[1] - prev_mouse_pos[1])
    return ActionRecord(
        timestamp=timestamp,
        mouse_dx=mouse_dx,
        mouse_dy=mouse_dy,
        mouse_left=input_state.mouse_left,
        mouse_right=input_state.mouse_right,
        keys=input_state.get_keys_snapshot(),
    )


def start_listeners(input_state: InputState):
    def on_press(key):
        try:
            if isinstance(key, keyboard.KeyCode) and key.char:
                input_state.pressed_keys.add(key.char.lower())
            else:
                input_state.pressed_keys.add(str(key))
        except Exception:
            pass

    def on_release(key):
        try:
            if isinstance(key, keyboard.KeyCode) and key.char:
                input_state.pressed_keys.discard(key.char.lower())
            else:
                input_state.pressed_keys.discard(str(key))
        except Exception:
            pass

    def on_click(x, y, button, pressed):
        if button == mouse.Button.left:
            input_state.mouse_left = 1 if pressed else 0
        elif button == mouse.Button.right:
            input_state.mouse_right = 1 if pressed else 0

    kb_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    ms_listener = mouse.Listener(on_click=on_click)
    kb_listener.start()
    ms_listener.start()
    return kb_listener, ms_listener


def _get_next_traj_index(h5_file: h5py.File) -> int:
    indices = []
    for name in h5_file.keys():
        if name.startswith("traj_"):
            try:
                indices.append(int(name.split("_")[1]))
            except (IndexError, ValueError):
                continue
    return max(indices) + 1 if indices else 0


def record_trajectory(output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    input_state = InputState()
    kb_listener, ms_listener = start_listeners(input_state)

    with mss.mss() as sct:
        # 改为全屏采集（主显示器）
        monitor = get_fullscreen_monitor(sct)
        frames: List[np.ndarray] = []  # 每帧 (90, 410, 3)
        actions: List[ActionRecord] = []

        print("准备录制：")
        print(f"- 目标输出文件: {output_path}")
        print(f"- 主画面 → {MAIN_TARGET_SIZE[0]}x{MAIN_TARGET_SIZE[1]}, 小地图 → {MINIMAP_TARGET_SIZE[0]}x{MINIMAP_TARGET_SIZE[1]}, 拼接 → (90, 410, 3)")
        print(f"- 目标帧率: {TARGET_FPS} FPS")
        print(f"- 按 F9 开始/暂停录制，按 F12 结束并保存。")

        control = {"recording": False, "exit": False}

        prev_time = time.time()
        prev_mouse_pos = None

        def on_control_key(key):
            # F9：开始/暂停录制
            if key == START_STOP_KEY:
                control["recording"] = not control["recording"]
                state = "开始录制" if control["recording"] else "暂停录制"
                print(f"[控制] {state}")
            # F12：退出
            elif key == EXIT_KEY:
                control["exit"] = True
                print("[控制] 收到退出指令，结束采集，请先不要结束程序")
                return False

        # 单独的控制按键监听
        with keyboard.Listener(on_press=on_control_key) as ctrl_listener:
            while not control["exit"]:
                now = time.time()
                if now - prev_time < FRAME_INTERVAL:
                    time.sleep(0.001)
                    continue
                prev_time = now

                raw = sct.grab(monitor)
                img = np.array(raw, dtype=np.uint8)
                img_main = crop_main_center_half(img, monitor)
                minimap_xywh = _minimap_center_to_roi_abs(monitor, MINIMAP_CENTER_REL)

                frame_main = preprocess_frame(img_main, MAIN_TARGET_SIZE)   # (90, 320, 3)
                frame_minimap = preprocess_frame_roi(img, minimap_xywh, MINIMAP_TARGET_SIZE)
                apply_circular_minimap_mask(frame_minimap)
                frame = np.hstack([frame_main, frame_minimap])  # (90, 410, 3)

                # 预览：拼接后整图
                disp = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # (90, 410, 3)
                if control["recording"]:
                    color = (0, 0, 255)
                    text = "REC"
                else:
                    color = (0, 255, 0)
                    text = "IDLE"
                cv2.circle(disp, (10, 10), 5, color, -1)
                cv2.putText(disp, text, (20, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                disp_s = cv2.resize(disp, (410 * 2, 90 * 2))
                cv2.putText(disp_s, "main|minimap", (5, disp_s.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                window_name = "COD_BC Capture"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(window_name, disp_s)

                # 尝试把小窗置顶，避免被全屏游戏遮住（仅在 Windows + pywin32 可用时生效）
                if win32gui is not None and win32con is not None:
                    try:
                        hwnd = win32gui.FindWindow(None, window_name)
                        if hwnd:
                            win32gui.SetWindowPos(
                                hwnd,
                                win32con.HWND_TOPMOST,
                                0,
                                0,
                                0,
                                0,
                                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW,
                            )
                    except Exception:
                        pass
                # 不使用键盘控制窗口，只是为了让窗口刷新
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    control["exit"] = True
                    print("[控制] 窗口关闭（q），结束采集")
                    break

                if not control["recording"]:
                    continue

                curr_mouse_pos = mouse.Controller().position
                action = encode_action(prev_mouse_pos, curr_mouse_pos, input_state, timestamp=now)
                prev_mouse_pos = curr_mouse_pos

                frames.append(frame)
                actions.append(action)

        kb_listener.stop()
        ms_listener.stop()
        cv2.destroyWindow("COD_BC Capture")

    # 如果一次都没开始录制（比如直接按 F12 退出），frames 会为空，
    # 此时不写入 h5 文件，直接返回，避免 np.stack 报错。
    if not frames:
        print("没有采集到任何帧，本次不保存轨迹。")
        return

    if len(frames) < TARGET_FPS * 60 * 5:
        print("Warning: trajectory shorter than 5 minutes, but saving anyway.")

    with h5py.File(output_path, "a") as f:
        traj_idx = _get_next_traj_index(f)
        grp = f.create_group(f"traj_{traj_idx}")
        frames_arr = np.stack(frames, axis=0)  # (N, 90, 410, 3)
        grp.create_dataset("states", data=frames_arr, compression="gzip")

        mouse_dx = np.array([a.mouse_dx for a in actions], dtype=np.float32)
        mouse_dy = np.array([a.mouse_dy for a in actions], dtype=np.float32)
        mouse_left = np.array([a.mouse_left for a in actions], dtype=np.int8)
        mouse_right = np.array([a.mouse_right for a in actions], dtype=np.int8)
        timestamps = np.array([a.timestamp for a in actions], dtype=np.float64)
        grp.create_dataset("mouse_dx", data=mouse_dx)
        grp.create_dataset("mouse_dy", data=mouse_dy)
        grp.create_dataset("mouse_left", data=mouse_left)
        grp.create_dataset("mouse_right", data=mouse_right)
        grp.create_dataset("timestamps", data=timestamps)

        # Save monitor / window metadata for later normalization
        grp.attrs["monitor_left"] = monitor["left"]
        grp.attrs["monitor_top"] = monitor["top"]
        grp.attrs["monitor_width"] = monitor["width"]
        grp.attrs["monitor_height"] = monitor["height"]
        grp.attrs["target_fps"] = TARGET_FPS

        dt = h5py.string_dtype(encoding="utf-8")
        max_len = max(len(a.keys) for a in actions) if actions else 0
        keys_matrix = []
        for a in actions:
            pad = a.keys + [""] * (max_len - len(a.keys))
            keys_matrix.append(pad)
        if keys_matrix:
            grp.create_dataset("keys", data=np.array(keys_matrix, dtype=dt))

    print(f"Saved trajectory with {len(frames)} frames to {output_path}")


def run_data_collection(output_path: str):
    record_trajectory(output_path)


if __name__ == "__main__":
    run_data_collection(os.path.join("data", "expert.h5"))

