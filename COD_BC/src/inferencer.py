# 该文件基于 https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning 的 dm_run_agent / key_output 思路改写
# 原作者：Tim Pearce, Jun Zhu
# 衍生项目：COD_BC (https://github.com/ImJX-Xu/COD_Behavioural_Cloning)，使用 pyautogui 控制
from typing import Optional

import time

import mss
import numpy as np
import pyautogui
import torch
from pynput import keyboard

from config import (
    TARGET_FPS,
    USE_EFFICIENTNET,
    USE_KEYBOARD,
    USE_MINIMAP,
    MOUSE_X_POSSIBLES,
    MOUSE_Y_POSSIBLES,
    IS_SPLIT_MOUSE,
    MAIN_TARGET_SIZE,
    MINIMAP_TARGET_SIZE,
    KEY_PROB_THRESHOLD,
    DEBUG_PRINT_MOUSE_INFER,
)
from .data_collector import (
    crop_main_center_half,
    MINIMAP_CENTER_REL,
    _minimap_center_to_roi_abs,
    get_fullscreen_monitor,
    preprocess_frame,
    preprocess_frame_roi,
    apply_circular_minimap_mask,
)
from .model import build_model


def load_model(checkpoint_path: str, device: torch.device):
    model = build_model(device)
    model.to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def run_inference(checkpoint_path: str, device: Optional[str] = None) -> None:
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, dev)

    frame_buffer_global = []
    frame_buffer_mm = []
    max_frames = 1 if USE_EFFICIENTNET else 4

    control = {"running": False, "stop": False}

    # 调试开关：FORCE_HOLD_W 无视模型输出，始终按住 W（用于验证键盘管道）
    FORCE_HOLD_W = False

    print("实时推理已启动：")
    print("- 按 F9 开始/暂停由模型控制键鼠")
    print("- 按 F12 立即停止并退出推理")

    def on_key_press(key):
        # F9：开关模型控制
        if key == keyboard.Key.f9:
            control["running"] = not control["running"]
            state = "开始由模型控制" if control["running"] else "暂停模型控制"
            print(f"[INFER] {state}")
        # F12：退出推理
        elif key == keyboard.Key.f12:
            control["stop"] = True
            print("[INFER] 收到退出指令，结束推理")
            return False

    listener = keyboard.Listener(on_press=on_key_press)
    listener.start()

    pyautogui.FAILSAFE = False
    pressed_keys = {"w": False, "a": False, "s": False, "d": False, "shift": False, "3": False}
    pressed_mouse = {"left": False, "right": False}

    # 推理时保持 LSTM 状态，实现跨步长时记忆（与 Example stateful 一致）
    lstm_state = None

    with mss.mss() as sct:
        # 与采集阶段保持一致：采集主显示器全屏（适配 COD 全屏无边框）
        monitor = get_fullscreen_monitor(sct)
        width = monitor["width"]
        height = monitor["height"]

        # 推理循环时间间隔：与采集/Example 一致（TARGET_FPS）
        target_interval = 1.0 / float(TARGET_FPS)
        last_time = time.time()

        while not control["stop"]:
            now = time.time()
            if now - last_time < target_interval:
                time.sleep(0.001)
                continue
            last_time = now

            raw = sct.grab(monitor)
            img = np.array(raw, dtype=np.uint8)
            img_main = crop_main_center_half(img, monitor)
            if USE_EFFICIENTNET:
                frame_main = preprocess_frame(img_main, MAIN_TARGET_SIZE)   # (90, 320, 3)
                mm_xywh = _minimap_center_to_roi_abs(monitor, MINIMAP_CENTER_REL)
                frame_mm = preprocess_frame_roi(img, mm_xywh, MINIMAP_TARGET_SIZE)
                apply_circular_minimap_mask(frame_mm)
                frame = np.hstack([frame_main, frame_mm])  # (90, 410, 3)
                frame_global = frame.astype("float32") / 255.0
                obs_np = np.transpose(frame_global, (2, 0, 1))  # (3, 90, 410)
            else:
                frame_global = preprocess_frame(img_main)
                frame_global = frame_global.astype("float32") / 255.0
                if USE_MINIMAP:
                    mm_xywh = _minimap_center_to_roi_abs(monitor, MINIMAP_CENTER_REL)
                    frame_mm = preprocess_frame_roi(img, mm_xywh).astype("float32") / 255.0
                else:
                    frame_mm = np.zeros_like(frame_global)
                frame_buffer_global.append(frame_global)
                frame_buffer_mm.append(frame_mm)
                if len(frame_buffer_global) > max_frames:
                    frame_buffer_global.pop(0)
                    frame_buffer_mm.pop(0)
                if len(frame_buffer_global) < max_frames:
                    continue
                stacked_global = np.stack(frame_buffer_global, axis=0)
                stacked_mm = np.stack(frame_buffer_mm, axis=0)
                obs_np = np.concatenate([stacked_global, stacked_mm], axis=0)

            # 如果当前处于暂停状态，只维护帧缓冲，不做任何键鼠操作，并清空 LSTM 状态
            if not control["running"]:
                lstm_state = None
                continue

            if USE_EFFICIENTNET:
                obs = torch.from_numpy(obs_np).unsqueeze(0).unsqueeze(0).to(dev)  # (1, 1, 3, H, W)
            else:
                obs = torch.from_numpy(obs_np).unsqueeze(0).unsqueeze(1).to(dev)

            with torch.no_grad():
                mouse_x_logits, mouse_y_logits, mouse_btn_logits, key_logits, lstm_state = model(
                    obs, lstm_state
                )

            # 离散鼠标：argmax 取档位，查表得像素位移
            idx_x = mouse_x_logits.squeeze(0).squeeze(0).argmax(dim=-1).item()
            idx_y = mouse_y_logits.squeeze(0).squeeze(0).argmax(dim=-1).item()
            dx = float(MOUSE_X_POSSIBLES[idx_x])
            dy = float(MOUSE_Y_POSSIBLES[idx_y])

            if DEBUG_PRINT_MOUSE_INFER:
                print(f"[INFER] mouse_bins=({idx_x},{idx_y}), move=({dx:.1f},{dy:.1f})")

            mouse_btn_probs = torch.sigmoid(mouse_btn_logits).squeeze(0).squeeze(0).cpu().numpy()  # [2]
            key_probs = torch.sigmoid(key_logits).squeeze(0).squeeze(0).cpu().numpy()  # [6]

            # IS_SPLIT_MOUSE：位移减半、应用两次，与 Example 一致，观感更顺
            if IS_SPLIT_MOUSE:
                pyautogui.moveRel(dx / 2.0, dy / 2.0, duration=0.0)
                time.sleep(0.015)  # 约半帧间隔
                pyautogui.moveRel(dx / 2.0, dy / 2.0, duration=0.0)
            else:
                pyautogui.moveRel(dx, dy, duration=0.0)

            # 鼠标左右键
            left_down = bool(mouse_btn_probs[0] > 0.5)
            right_down = bool(mouse_btn_probs[1] > 0.5)

            if left_down and not pressed_mouse["left"]:
                pyautogui.mouseDown(button="left")
                pressed_mouse["left"] = True
            elif (not left_down) and pressed_mouse["left"]:
                pyautogui.mouseUp(button="left")
                pressed_mouse["left"] = False

            if right_down and not pressed_mouse["right"]:
                pyautogui.mouseDown(button="right")
                pressed_mouse["right"] = True
            elif (not right_down) and pressed_mouse["right"]:
                pyautogui.mouseUp(button="right")
                pressed_mouse["right"] = False

            # 键盘：两种模式
            if FORCE_HOLD_W:
                # 调试模式：无视模型输出，始终按住 W，确认 COD 能接受本进程发出的按键。
                if not pressed_keys["w"]:
                    pyautogui.keyDown("w")
                    pressed_keys["w"] = True
            elif USE_KEYBOARD:
                # 正常模式：根据模型预测的多标签概率控制按键。
                # 键顺序：w,a,s,d,shift,3
                keys = ["w", "a", "s", "d", "shift", "3"]
                for i, k in enumerate(keys):
                    want_down = bool(key_probs[i] > KEY_PROB_THRESHOLD)
                    if want_down and not pressed_keys[k]:
                        pyautogui.keyDown(k)
                        pressed_keys[k] = True
                    elif (not want_down) and pressed_keys[k]:
                        pyautogui.keyUp(k)
                        pressed_keys[k] = False

    # 退出时确保释放按键
    for k, is_down in pressed_keys.items():
        if is_down:
            try:
                pyautogui.keyUp(k)
            except Exception:
                pass
    for btn, is_down in pressed_mouse.items():
        if is_down:
            try:
                pyautogui.mouseUp(button=btn)
            except Exception:
                pass


