from typing import Optional
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import (
    USE_KEYBOARD,
    N_MOUSE_X,
    N_MOUSE_Y,
    BATCH_SIZE,
    ACCUM_STEPS,
    LR,
    WEIGHT_DECAY,
    LR_STEP_SIZE,
    LR_GAMMA,
    EARLY_STOP_PATIENCE,
    MAX_EPOCHS,
    TRAIN_KEYS_WASD,
    TRAIN_KEYS_SHIFT3,
    LOSS_WEIGHT_MOUSE_NO_KEY,
    LOSS_WEIGHT_BTN_NO_KEY,
    LOSS_WEIGHT_MOUSE_WITH_KEY,
    LOSS_WEIGHT_BTN_WITH_KEY,
    LOSS_WEIGHT_KEYS_WITH_KEY,
)
from .data_processor import create_splits
from .model import build_model

# 让 cuDNN 为当前输入尺寸搜索最快卷积实现（首次会略慢，之后更快）
torch.backends.cudnn.benchmark = True

try:
    import pynvml  # type: ignore

    _HAS_PYNVML = True
    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    _HAS_PYNVML = False
    _NVML_HANDLE = None


def _get_cuda_stats(device: torch.device):
    """
    返回当前 CUDA 显存 / 利用率情况（如果可用）。
    利用率需要系统已安装 pynvml，否则只返回显存信息。
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return None

    mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
    mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)

    stats = {
        "mem_alloc_mb": mem_alloc,
        "mem_reserved_mb": mem_reserved,
    }

    if _HAS_PYNVML and _NVML_HANDLE is not None:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
            stats["util_percent"] = util.gpu
        except Exception:
            pass

    return stats


def run_training(data_path: str, num_epochs: int = 200, checkpoint_dir: Optional[str] = None) -> None:
    if not os.path.exists(data_path):
        print(f"找不到数据集文件: {data_path}")
        print("请先执行数据采集，再进行训练：")
        print("  1) 运行 `python main.py` 选择 1 进行数据采集")
        print("  2) 采集完成后再选择 2 进行模型训练")
        return

    if checkpoint_dir is None:
        checkpoint_dir = "models"
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from config import SEQ_LEN
    train_ds, val_ds, _ = create_splits(data_path, seq_len=SEQ_LEN, frame_stack=4)
    if train_ds is None or len(train_ds) == 0:
        raise RuntimeError("No training data found in H5 file.")

    # 与 Example 一致：小 batch + 梯度累积；intermediate backbone 已减显存
    batch_size = BATCH_SIZE
    accum_steps = max(1, ACCUM_STEPS)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0) if val_ds else None

    # 从第一个样本推断当前状态空间/动作空间形状（避免手写硬编码）
    sample = train_ds[0]
    obs_shape = tuple(sample["obs"].shape)               # (T, C, H, W)
    mouse_btn_shape = tuple(sample["mouse_buttons"].shape)  # (T, 2)
    keys_shape = tuple(sample["keys"].shape)             # (T, 6)

    print(f"[TRAIN] Using device: {device}, batch_size={batch_size}, accum_steps={accum_steps}")
    print(f"[TRAIN] Train sequences: {len(train_ds)}, batches per epoch: {len(train_loader)}")
    if val_loader:
        print(f"[TRAIN] Val sequences: {len(val_ds)}, batches per epoch: {len(val_loader)}")
    else:
        print("[TRAIN] No separate validation set, will use train loss as val loss.")

    print(f"[TRAIN] State space: obs[T,C,H,W]={obs_shape}")
    print(
        "[TRAIN] Action space: "
        f"mouse_x_classes={N_MOUSE_X}, mouse_y_classes={N_MOUSE_Y}, "
        f"mouse_buttons_shape={mouse_btn_shape}, keys_shape={keys_shape}, "
        f"USE_KEYBOARD={USE_KEYBOARD}, TRAIN_KEYS_WASD={TRAIN_KEYS_WASD}, TRAIN_KEYS_SHIFT3={TRAIN_KEYS_SHIFT3}"
    )

    print("[TRAIN] Building model (EfficientNet + ConvLSTM/LSTM)...")
    model = build_model(device)
    print("[TRAIN] Model ready, start training.")

    ce_loss = nn.CrossEntropyLoss(reduction="mean")
    bce_loss = nn.BCEWithLogitsLoss()
    # 键盘损失按键位逐维计算，mask 由 config 中的 TRAIN_KEYS_WASD / TRAIN_KEYS_SHIFT3 控制
    bce_keys = nn.BCEWithLogitsLoss(reduction="none")
    key_loss_mask_vals = [
        1 if TRAIN_KEYS_WASD else 0,    # w
        1 if TRAIN_KEYS_WASD else 0,    # a
        1 if TRAIN_KEYS_WASD else 0,    # s
        1 if TRAIN_KEYS_WASD else 0,    # d
        1 if TRAIN_KEYS_SHIFT3 else 0,  # shift
        1 if TRAIN_KEYS_SHIFT3 else 0,  # 3
    ]
    use_key_loss = any(v == 1 for v in key_loss_mask_vals)
    key_loss_mask_tensor = torch.tensor(key_loss_mask_vals, dtype=torch.float32, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    writer = SummaryWriter(log_dir="logs")

    best_val_loss = float("inf")
    epochs_no_improve = 0
    early_stop_patience = EARLY_STOP_PATIENCE

    global_step = 0
    effective_epochs = min(num_epochs, MAX_EPOCHS)
    if num_epochs > MAX_EPOCHS:
        print(f"[TRAIN] 要求训练 {num_epochs} 个 epoch，但已被 config.MAX_EPOCHS={MAX_EPOCHS} 截断为 {effective_epochs}。")

    for epoch in range(1, effective_epochs + 1):
        print(f"[TRAIN] ===== Epoch {epoch}/{effective_epochs} =====")
        model.train()
        train_loss_sum = 0.0

        optimizer.zero_grad()
        accum_count = 0
        for batch_idx, batch in enumerate(train_loader, start=1):
            obs = batch["obs"].to(device)
            mouse_x_tgt = batch["mouse_x_class"].to(device)  # [B, T]
            mouse_y_tgt = batch["mouse_y_class"].to(device)
            mouse_btn_tgt = batch["mouse_buttons"].to(device)
            keys_tgt = batch["keys"].to(device)

            mouse_x_logits, mouse_y_logits, mouse_btn_logits, key_logits, _ = model(obs)

            # 鼠标：分类交叉熵（flatten 成 N, C 与 N,）
            B, T = mouse_x_tgt.shape[0], mouse_x_tgt.shape[1]
            loss_mouse_x = ce_loss(
                mouse_x_logits.view(B * T, N_MOUSE_X), mouse_x_tgt.view(B * T)
            )
            loss_mouse_y = ce_loss(
                mouse_y_logits.view(B * T, N_MOUSE_Y), mouse_y_tgt.view(B * T)
            )
            mouse_btn_loss = bce_loss(mouse_btn_logits, mouse_btn_tgt)
            if USE_KEYBOARD and use_key_loss:
                # 仅对配置中启用的键位计算 BCE 损失（例如默认只训 WASD，忽略 Shift/3）
                key_loss_raw = bce_keys(key_logits, keys_tgt)  # [B, T, 6]
                keys_loss = (key_loss_raw * key_loss_mask_tensor).sum(dim=-1).mean()
                loss = LOSS_WEIGHT_MOUSE_WITH_KEY * (loss_mouse_x + loss_mouse_y) + LOSS_WEIGHT_BTN_WITH_KEY * mouse_btn_loss + LOSS_WEIGHT_KEYS_WITH_KEY * keys_loss
            else:
                loss = LOSS_WEIGHT_MOUSE_NO_KEY * (loss_mouse_x + loss_mouse_y) + LOSS_WEIGHT_BTN_NO_KEY * mouse_btn_loss
            (loss / accum_steps).backward()
            accum_count += 1

            train_loss_sum += loss.item()
            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

            # 梯度累积：累计满 accum_steps（或到最后一个 batch）再更新一次
            if accum_count >= accum_steps or batch_idx == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                accum_count = 0

            # 训练过程打印：每 10 个 batch 或最后一个 batch 打一行
            if batch_idx % 10 == 0 or batch_idx == len(train_loader):
                msg = (
                    f"[TRAIN][epoch {epoch}] "
                    f"batch {batch_idx}/{len(train_loader)}, "
                    f"loss={loss.item():.4f}"
                )

                cuda_stats = _get_cuda_stats(device)
                if cuda_stats is not None:
                    util_part = ""
                    if "util_percent" in cuda_stats:
                        util_part = f", util={cuda_stats['util_percent']:3d}%"
                    msg += (
                        f" | cuda_mem={cuda_stats['mem_alloc_mb']:.1f}"
                        f"/{cuda_stats['mem_reserved_mb']:.1f} MB"
                        f"{util_part}"
                    )

                print(msg)

        scheduler.step()
        avg_train_loss = train_loss_sum / max(len(train_loader), 1)

        if val_loader:
            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader, start=1):
                    obs = batch["obs"].to(device)
                    mouse_x_tgt = batch["mouse_x_class"].to(device)
                    mouse_y_tgt = batch["mouse_y_class"].to(device)
                    mouse_btn_tgt = batch["mouse_buttons"].to(device)
                    keys_tgt = batch["keys"].to(device)

                    mouse_x_logits, mouse_y_logits, mouse_btn_logits, key_logits, _ = model(obs)

                    B, T = mouse_x_tgt.shape[0], mouse_x_tgt.shape[1]
                    loss_mouse_x = ce_loss(
                        mouse_x_logits.view(B * T, N_MOUSE_X), mouse_x_tgt.view(B * T)
                    )
                    loss_mouse_y = ce_loss(
                        mouse_y_logits.view(B * T, N_MOUSE_Y), mouse_y_tgt.view(B * T)
                    )
                    mouse_btn_loss = bce_loss(mouse_btn_logits, mouse_btn_tgt)
                    if USE_KEYBOARD and use_key_loss:
                        key_loss_raw = bce_keys(key_logits, keys_tgt)  # [B, T, 6]
                        keys_loss = (key_loss_raw * key_loss_mask_tensor).sum(dim=-1).mean()
                        loss = LOSS_WEIGHT_MOUSE_WITH_KEY * (loss_mouse_x + loss_mouse_y) + LOSS_WEIGHT_BTN_WITH_KEY * mouse_btn_loss + LOSS_WEIGHT_KEYS_WITH_KEY * keys_loss
                    else:
                        loss = LOSS_WEIGHT_MOUSE_NO_KEY * (loss_mouse_x + loss_mouse_y) + LOSS_WEIGHT_BTN_NO_KEY * mouse_btn_loss
                    val_loss_sum += loss.item()

                    # 验证过程简单打印最后一个 batch 的 loss
                    if batch_idx == len(val_loader):
                        print(
                            f"[VAL][epoch {epoch}] "
                            f"batch {batch_idx}/{len(val_loader)}, "
                            f"loss={loss.item():.4f}"
                        )

            avg_val_loss = val_loss_sum / max(len(val_loader), 1)
        else:
            avg_val_loss = avg_train_loss

        writer.add_scalar("epoch/train_loss", avg_train_loss, epoch)
        writer.add_scalar("epoch/val_loss", avg_val_loss, epoch)
        writer.add_scalar("epoch/lr", scheduler.get_last_lr()[0], epoch)

        print(
            f"[EPOCH {epoch}/{num_epochs}] "
            f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                },
                os.path.join(checkpoint_dir, "best_model.pt"),
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                break

    writer.close()


