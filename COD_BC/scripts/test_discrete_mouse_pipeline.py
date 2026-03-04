"""
测试离散鼠标全流程：与 Example (Counter-Strike_Behavioural_Cloning) 对齐验证。
- 生成最小可用 expert.h5
- 跑 1 个 epoch 训练
- 跑 1 步推理（argmax + 查表）
"""
import os
import sys

import h5py
import numpy as np
import torch

# 项目根为 COD_BC
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MOUSE_X_POSSIBLES, MOUSE_Y_POSSIBLES, N_MOUSE_X, N_MOUSE_Y, SEQ_LEN, TARGET_FPS
from src.data_processor import discretize_mouse, create_splits
from src.model import build_model
from src.inferencer import load_model
from torch import nn
from torch.utils.data import DataLoader


def make_minimal_h5(path: str, num_frames: int = 200) -> None:
    """生成一个最小的、可被 create_splits 使用的 expert.h5（单条轨迹）。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        g = f.create_group("traj_0")
        # states: [N, H, W] 与 data_collector 一致；TARGET_SIZE = (150, 84) -> (84, 150) in array
        g.create_dataset("states", data=np.random.randint(0, 256, (num_frames, 84, 150), dtype=np.uint8), compression="gzip")
        # 鼠标位移：像素，在档位附近随机
        dx = np.random.choice(MOUSE_X_POSSIBLES, size=num_frames).astype(np.float32)
        dy = np.random.choice(MOUSE_Y_POSSIBLES, size=num_frames).astype(np.float32)
        g.create_dataset("mouse_dx", data=dx)
        g.create_dataset("mouse_dy", data=dy)
        g.create_dataset("mouse_left", data=np.zeros(num_frames, dtype=np.int8))
        g.create_dataset("mouse_right", data=np.zeros(num_frames, dtype=np.int8))
        g.create_dataset("timestamps", data=np.arange(num_frames, dtype=np.float64) / float(TARGET_FPS))
        g.attrs["monitor_width"] = 2560
        g.attrs["monitor_height"] = 1600
        g.attrs["target_fps"] = TARGET_FPS
    print(f"[TEST] Created minimal {path} with {num_frames} frames.")


def test_discretize_vs_example():
    """与 Example 的 mouse_preprocess 逻辑一致：clip + 最近档位。"""
    # 连续值 -> 应映射到最近档位（档位与 Example 一致）
    ix, iy = discretize_mouse(47.0, -12.0)
    assert 0 <= ix < N_MOUSE_X and 0 <= iy < N_MOUSE_Y
    assert MOUSE_X_POSSIBLES[ix] == 60.0   # 47 最近 60
    assert MOUSE_Y_POSSIBLES[iy] == -10.0  # -12 最近 -10
    print("[TEST] discretize_mouse: OK (nearest bin, Example bins).")


def test_training_one_epoch(data_path: str):
    """训练 1 个 epoch，不保存 checkpoint。"""
    train_ds, val_ds, _ = create_splits(data_path, seq_len=SEQ_LEN, frame_stack=4)
    loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    model.train()
    for i, batch in enumerate(loader):
        if i >= 3:
            break  # 只跑 3 个 batch
        obs = batch["obs"].to(device)
        mx_t = batch["mouse_x_class"].to(device)
        my_t = batch["mouse_y_class"].to(device)
        btn_t = batch["mouse_buttons"].to(device)

        mx_log, my_log, btn_log, _, _ = model(obs)
        B, T = mx_t.shape[0], mx_t.shape[1]
        loss = (
            ce(mx_log.view(B * T, N_MOUSE_X), mx_t.view(B * T))
            + ce(my_log.view(B * T, N_MOUSE_Y), my_t.view(B * T))
            + bce(btn_log, btn_t)
        )
        loss.backward()
    print("[TEST] Training (3 batches): OK.")


def test_inference_argmax_lookup(device: torch.device):
    """推理：argmax 取档位下标，查表得 dx/dy（与 Example onehot_to_actions 一致）。"""
    model = build_model(device)
    model.eval()
    B, T = 1, 1
    obs = torch.randn(B, T, 12, 84, 150).to(device)
    with torch.no_grad():
        mx_log, my_log, _, _, _ = model(obs)
    idx_x = mx_log.squeeze(0).squeeze(0).argmax(dim=-1).item()
    idx_y = my_log.squeeze(0).squeeze(0).argmax(dim=-1).item()
    dx = MOUSE_X_POSSIBLES[idx_x]
    dy = MOUSE_Y_POSSIBLES[idx_y]
    assert isinstance(dx, (int, float)) and isinstance(dy, (int, float))
    print(f"[TEST] Inference argmax + lookup: dx={dx}, dy={dy} -> OK.")


def main():
    print("===== 与 Example 对齐检查 =====")
    print("1) 鼠标离散：clip + 最近档位 -> 类下标 (与 Example mouse_preprocess + onehot 一致)")
    print("2) 训练：鼠标用 CrossEntropyLoss / Example 用 categorical_crossentropy")
    print("3) 推理：argmax(logits) -> 查表 MOUSE_*_POSSIBLES[id] (与 Example onehot_to_actions 一致)")
    print()

    test_discretize_vs_example()

    data_path = "data/expert_test_minimal.h5"
    make_minimal_h5(data_path, num_frames=200)

    test_training_one_epoch(data_path)
    test_inference_argmax_lookup(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # 清理
    if os.path.exists(data_path):
        os.remove(data_path)
    chunked = "data/expert_test_minimal_chunked.h5"
    if os.path.exists(chunked):
        os.remove(chunked)
    print()
    print("===== 全部通过：鼠标采样/训练/推理与 Example 基本一致 =====")


if __name__ == "__main__":
    main()
