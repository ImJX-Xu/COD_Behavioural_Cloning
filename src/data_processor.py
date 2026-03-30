import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# 功能开关与鼠标档位：从总控 config 读取
from config import (
    SEQ_LEN,
    USE_KEYBOARD,
    USE_MINIMAP,
    MOUSE_X_POSSIBLES,
    MOUSE_Y_POSSIBLES,
    CHUNK_MIN_FRAMES,
    CHUNK_MAX_FRAMES,
)

ACTION_KEYS = ("w", "a", "s", "d", "shift", "3")

REVERSE_KEY_MAP = {
    "Key.shift": "shift",
    "Key.shift_r": "shift",
    "Key.ctrl": "ctrl",
    "Key.ctrl_r": "ctrl",
    "Key.space": "space",
}


def normalize_key(k: str) -> str:
    k = k.lower()
    return REVERSE_KEY_MAP.get(k, k)


def encode_keyboard_multi_hot(keys: Sequence[str]) -> np.ndarray:
    norm = {normalize_key(k) for k in keys}
    out = np.zeros((len(ACTION_KEYS),), dtype="float32")
    for i, k in enumerate(ACTION_KEYS):
        out[i] = 1.0 if k in norm else 0.0
    return out


def discretize_mouse(dx: float, dy: float) -> Tuple[int, int]:
    """将连续像素位移映射到最近档位，返回 (bin_x, bin_y) 类下标。"""
    dx = max(min(dx, MOUSE_X_POSSIBLES[-1]), MOUSE_X_POSSIBLES[0])
    dy = max(min(dy, MOUSE_Y_POSSIBLES[-1]), MOUSE_Y_POSSIBLES[0])
    ix = int(np.argmin(np.abs(np.array(MOUSE_X_POSSIBLES, dtype="float32") - dx)))
    iy = int(np.argmin(np.abs(np.array(MOUSE_Y_POSSIBLES, dtype="float32") - dy)))
    return ix, iy


@dataclass
class SequenceSampleIndex:
    traj_name: str
    start_idx: int


def _chunk_trajectory(
    num_frames: int,
    seq_len: int,
    chunk_min: int = CHUNK_MIN_FRAMES,
    chunk_max: int = CHUNK_MAX_FRAMES,
) -> list[Tuple[int, int]]:
    """
    将长轨迹切割为 20–30 秒的段，返回 [(start, end), ...]。
    保证每个 chunk 至少有 seq_len 帧以便采样。
    """
    if num_frames < seq_len:
        return []
    chunks: list[Tuple[int, int]] = []
    pos = 0
    while pos < num_frames:
        remain = num_frames - pos
        size = min(chunk_max, max(chunk_min, remain))
        end = min(pos + size, num_frames)
        if end - pos >= seq_len:
            chunks.append((pos, end))
        pos = end
    return chunks


def build_chunked_h5(
    src_path: str,
    dst_path: str,
    chunk_min: int = CHUNK_MIN_FRAMES,
    chunk_max: int = CHUNK_MAX_FRAMES,
    seq_len: int = SEQ_LEN,
) -> str:
    """
    从原始 expert.h5 切割轨迹为 20–30 秒段，写入新文件 expert_chunked.h5。
    不修改原文件。返回 dst_path。
    """
    chunk_idx = 0
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for name in sorted(src.keys()):
            if not name.startswith("traj_"):
                continue
            grp_src = src[name]
            if "states" not in grp_src:
                continue
            num_frames = grp_src["states"].shape[0]
            chunks = _chunk_trajectory(num_frames, seq_len, chunk_min, chunk_max)
            if not chunks:
                chunks = [(0, num_frames)]
            for start, end in chunks:
                gname = f"chunk_{chunk_idx}"
                grp_dst = dst.create_group(gname)
                grp_dst.create_dataset("states", data=grp_src["states"][start:end], compression="gzip")
                if "states_minimap" in grp_src:
                    grp_dst.create_dataset("states_minimap", data=grp_src["states_minimap"][start:end], compression="gzip")
                grp_dst.create_dataset("mouse_dx", data=grp_src["mouse_dx"][start:end])
                grp_dst.create_dataset("mouse_dy", data=grp_src["mouse_dy"][start:end])
                grp_dst.create_dataset("mouse_left", data=grp_src["mouse_left"][start:end])
                grp_dst.create_dataset("mouse_right", data=grp_src["mouse_right"][start:end])
                grp_dst.create_dataset("timestamps", data=grp_src["timestamps"][start:end])
                for k, v in grp_src.attrs.items():
                    grp_dst.attrs[k] = v
                if "keys" in grp_src:
                    grp_dst.create_dataset("keys", data=grp_src["keys"][start:end])
                chunk_idx += 1
    return dst_path


def _get_chunked_path(h5_path: str) -> str:
    base, ext = os.path.splitext(h5_path)
    return f"{base}_chunked{ext}"


class H5SequenceDataset(Dataset):
    """
    Dataset of fixed-length sequences for LSTM-BC training.
    期望输入为切割后的 chunked h5 文件，每条轨迹已在 20–30 秒范围内。

    USE_EFFICIENTNET 时：每步单帧 RGB，obs [seq_len, 3, H, W]（与 Example 一致）。
    否则：4 帧堆叠 + 主/小地图，obs [seq_len, 8, 84, 84] 或 [seq_len, 4, 84, 84]。

    Each item:
      - obs: [seq_len, C, H, W], float32 in [0,1]
      - mouse_x_class, mouse_y_class, mouse_buttons, keys
    """

    def __init__(
        self,
        h5_path: str,
        traj_names: Sequence[str],
        seq_len: int = SEQ_LEN,
        frame_stack: int = 4,
    ):
        super().__init__()
        self.h5_path = h5_path
        self.traj_names = list(traj_names)
        self.seq_len = seq_len
        # frame_stack 仅用于旧灰度堆叠模式，现已废弃，保留参数以保持接口兼容。

        self._indices: List[SequenceSampleIndex] = []

        with h5py.File(self.h5_path, "r") as f:
            for name in self.traj_names:
                grp = f[name]
                states_shape = grp["states"].shape
                num_frames = states_shape[0]
                need = self.seq_len
                max_start = num_frames - need
                if max_start < 0:
                    continue
                for start in range(0, max_start + 1):
                    self._indices.append(SequenceSampleIndex(traj_name=name, start_idx=start))

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        meta = self._indices[idx]
        with h5py.File(self.h5_path, "r") as f:
            grp = f[meta.traj_name]
            states = grp["states"]
            mouse_dx = grp["mouse_dx"]
            mouse_dy = grp["mouse_dy"]
            mouse_left = grp["mouse_left"]
            mouse_right = grp["mouse_right"]
            keys_ds = grp.get("keys")

            s = meta.start_idx
            e = s + self.seq_len

            # RGB 序列：obs [T, 3, H, W]
            frames = np.asarray(states[s:e], dtype=np.float32) / 255.0
            obs = np.transpose(frames, (0, 3, 1, 2)).astype("float32")  # (T,H,W,3) -> (T,3,H,W)

            # 离散化鼠标：每帧 (dx, dy) -> (bin_x, bin_y) 类下标
            mouse_x_class = np.zeros(self.seq_len, dtype=np.int64)
            mouse_y_class = np.zeros(self.seq_len, dtype=np.int64)
            for t in range(self.seq_len):
                ix, iy = discretize_mouse(float(mouse_dx[s + t]), float(mouse_dy[s + t]))
                mouse_x_class[t] = ix
                mouse_y_class[t] = iy

            mb = np.stack([mouse_left[s:e], mouse_right[s:e]], axis=-1).astype("float32")

            keys_vec = np.zeros((self.seq_len, len(ACTION_KEYS)), dtype="float32")
            if USE_KEYBOARD and keys_ds is not None:
                for t in range(self.seq_len):
                    raw_keys = keys_ds[s + t]
                    valid_keys = [k for k in raw_keys.tolist() if k]
                    keys_vec[t] = encode_keyboard_multi_hot(valid_keys)

        obs_t = torch.from_numpy(obs)
        mouse_x_class_t = torch.from_numpy(mouse_x_class)  # [T]
        mouse_y_class_t = torch.from_numpy(mouse_y_class)  # [T]
        mb_t = torch.from_numpy(mb)
        keys_t = torch.from_numpy(keys_vec)

        return {
            "obs": obs_t,  # [T, C, H, W]：当前为 RGB (T,3,90,410)
            "mouse_x_class": mouse_x_class_t,  # [T] 类下标
            "mouse_y_class": mouse_y_class_t,  # [T] 类下标
            "mouse_buttons": mb_t,  # [T, 2]
            "keys": keys_t,  # [T, 6]
        }


def create_splits(
    h5_path: str,
    seq_len: int = SEQ_LEN,
    frame_stack: int = 4,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
):
    chunked_path = _get_chunked_path(h5_path)
    need_rebuild = not os.path.exists(chunked_path)
    if not need_rebuild and os.path.exists(h5_path):
        need_rebuild = os.path.getmtime(h5_path) > os.path.getmtime(chunked_path)
    if need_rebuild:
        build_chunked_h5(h5_path, chunked_path, seq_len=seq_len)

    with h5py.File(chunked_path, "r") as f:
        traj_names = sorted([name for name in f.keys() if name.startswith("chunk_")])

    if not traj_names:
        raise FileNotFoundError(
            f"切割后无有效轨迹。请确认 {h5_path} 中含有 states 数据集的 traj_* 轨迹组。"
        )

    n = len(traj_names)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio)) if n > 2 else 0
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0

    train_trajs = traj_names[:n_train]
    val_trajs = traj_names[n_train : n_train + n_val]
    test_trajs = traj_names[n_train + n_val :]

    train_ds = H5SequenceDataset(chunked_path, train_trajs, seq_len=seq_len, frame_stack=frame_stack)
    val_ds = H5SequenceDataset(chunked_path, val_trajs, seq_len=seq_len, frame_stack=frame_stack) if val_trajs else None
    test_ds = H5SequenceDataset(chunked_path, test_trajs, seq_len=seq_len, frame_stack=frame_stack) if test_trajs else None

    return train_ds, val_ds, test_ds

