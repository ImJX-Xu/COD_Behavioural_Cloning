from typing import Optional

import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data_processor import create_splits
from .model import build_model


def _load_model(checkpoint_path: str, dev: torch.device):
    model = build_model(dev)
    ckpt = torch.load(checkpoint_path, map_location=dev)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def run_evaluation(data_path: str, checkpoint_path: str, device: Optional[str] = None) -> None:
    """
    离线评估当前模型在数据集上的模仿效果。

    指标包括：
      - 鼠标 x/y 分类准确率（argmax 与离散标签一致）
      - 鼠标左右键准确率
      - 键盘多标签准确率（W/A/S/D/Shift/3 六键）
      - 完整动作匹配率（鼠标键 + 键盘六键全部一致）
    """
    if not os.path.exists(data_path):
        print(f"[EVAL] 找不到数据文件: {data_path}，请先进行数据采集。")
        return
    if not os.path.exists(checkpoint_path):
        print(f"[EVAL] 找不到模型文件: {checkpoint_path}，请先完成训练。")
        return

    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 仅使用测试集进行评估；如果没有单独测试集，则退化为使用训练集
    print("[EVAL] 加载数据与划分训练/验证/测试集...")
    from config import SEQ_LEN
    train_ds, val_ds, test_ds = create_splits(data_path, seq_len=SEQ_LEN, frame_stack=4)
    eval_ds = test_ds or train_ds
    if eval_ds is None or len(eval_ds) == 0:
        print("[EVAL] 没有可用的数据用于评估。")
        return

    loader = DataLoader(eval_ds, batch_size=32, shuffle=False, num_workers=0)

    print(f"[EVAL] 使用设备: {dev}")
    print(f"[EVAL] 评估序列数: {len(eval_ds)}, 批次数: {len(loader)}")
    print("[EVAL] 加载模型权重...")
    model = _load_model(checkpoint_path, dev)

    mouse_btn_correct = 0
    mouse_btn_total = 0

    key_correct = 0
    key_total = 0

    full_action_correct = 0
    full_action_total = 0

    print("[EVAL] 开始前向推理与指标统计...")
    mouse_x_correct = 0
    mouse_x_total = 0
    mouse_y_correct = 0
    mouse_y_total = 0

    with torch.no_grad():
        for i, batch in enumerate(loader, start=1):
            obs = batch["obs"].to(dev)
            mouse_x_tgt = batch["mouse_x_class"].to(dev)  # [B,T]
            mouse_y_tgt = batch["mouse_y_class"].to(dev)
            mouse_btn_tgt = batch["mouse_buttons"].to(dev)  # [B,T,2]
            keys_tgt = batch["keys"].to(dev)  # [B,T,6]

            mouse_x_logits, mouse_y_logits, mouse_btn_logits, key_logits, _ = model(obs)

            # 鼠标 x/y：分类准确率（argmax 与标签一致）
            mouse_x_pred = mouse_x_logits.argmax(dim=-1)  # [B,T]
            mouse_y_pred = mouse_y_logits.argmax(dim=-1)
            mouse_x_correct += int((mouse_x_pred == mouse_x_tgt).sum().cpu().item())
            mouse_x_total += mouse_x_tgt.numel()
            mouse_y_correct += int((mouse_y_pred == mouse_y_tgt).sum().cpu().item())
            mouse_y_total += mouse_y_tgt.numel()

            # 鼠标左右键准确率
            mouse_btn_pred = (torch.sigmoid(mouse_btn_logits) > 0.5).float()
            mouse_btn_eq = (mouse_btn_pred == mouse_btn_tgt).all(dim=-1)  # [B,T]
            mouse_btn_correct += int(mouse_btn_eq.sum().cpu().item())
            mouse_btn_total += int(mouse_btn_eq.numel())

            # 键盘多标签准确率（逐帧全部 6 键是否完全一致）
            key_pred = (torch.sigmoid(key_logits) > 0.5).float()
            key_eq = (key_pred == keys_tgt).all(dim=-1)  # [B,T]
            key_correct += int(key_eq.sum().cpu().item())
            key_total += int(key_eq.numel())

            # 完整动作匹配：鼠标左右键 + 键盘六键全部一致
            full_eq = mouse_btn_eq & key_eq  # [B,T]
            full_action_correct += int(full_eq.sum().cpu().item())
            full_action_total += int(full_eq.numel())

            if i % 20 == 0 or i == len(loader):
                print(f"[EVAL] 进度: {i}/{len(loader)} 批次已完成")

    mouse_x_acc = mouse_x_correct / mouse_x_total if mouse_x_total > 0 else 0.0
    mouse_y_acc = mouse_y_correct / mouse_y_total if mouse_y_total > 0 else 0.0
    mouse_btn_acc = mouse_btn_correct / mouse_btn_total if mouse_btn_total > 0 else 0.0
    key_acc = key_correct / key_total if key_total > 0 else 0.0
    full_action_acc = full_action_correct / full_action_total if full_action_total > 0 else 0.0

    print("========== EVAL RESULT ==========")
    print(f"数据集: {data_path}")
    print(f"模型:   {checkpoint_path}")
    print(f"样本序列数: {len(eval_ds)}")
    print("---------------------------------")
    print(f"鼠标 X 分类准确率: {mouse_x_acc*100:.2f}%")
    print(f"鼠标 Y 分类准确率: {mouse_y_acc*100:.2f}%")
    print(f"鼠标左右键准确率: {mouse_btn_acc*100:.2f}%")
    print(f"键盘多标签准确率(WASD+Shift+3): {key_acc*100:.2f}%")
    print(f"完整动作匹配率:                 {full_action_acc*100:.2f}%")
    print("=================================")


