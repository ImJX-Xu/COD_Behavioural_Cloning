"""
高优先级修复回归测试（轻量）：
1) 鼠标离散化边界/最近档位
2) 小地图圆形遮罩输出
3) 推理模型加载的 checkpoint 存在性校验
"""
import os
import sys

import numpy as np
import torch

# 项目根为 BC_COD（scripts/ 的上一级）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from config import MOUSE_X_POSSIBLES, MOUSE_Y_POSSIBLES
from src.data_collector import apply_circular_minimap_mask
from src.data_processor import discretize_mouse
from src.inferencer import load_model


def test_discretize_mouse_nearest_bin() -> None:
    # 47.0 最近档位应为 44.6（当前 MOUSE_X_POSSIBLES 最大非极值档）
    # -12.0 最近档位应为 -9.0
    ix, iy = discretize_mouse(47.0, -12.0)
    assert MOUSE_X_POSSIBLES[ix] == 44.6, f"unexpected x bin: {MOUSE_X_POSSIBLES[ix]}"
    assert MOUSE_Y_POSSIBLES[iy] == -9.0, f"unexpected y bin: {MOUSE_Y_POSSIBLES[iy]}"


def test_discretize_mouse_clip_bound() -> None:
    ix, iy = discretize_mouse(10_000.0, -10_000.0)
    assert ix == len(MOUSE_X_POSSIBLES) - 1
    assert iy == 0


def test_circular_minimap_mask() -> None:
    img = np.ones((90, 90, 3), dtype=np.uint8) * 255
    out = apply_circular_minimap_mask(img.copy())

    # 四角应被遮罩为黑，中心应保留
    assert out[0, 0].sum() == 0
    assert out[0, 89].sum() == 0
    assert out[89, 0].sum() == 0
    assert out[89, 89].sum() == 0
    assert out[45, 45].sum() > 0


def test_load_model_checkpoint_not_found() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    missing_path = "models/this_file_should_not_exist.pt"
    try:
        load_model(missing_path, dev)
    except FileNotFoundError:
        return
    raise AssertionError("load_model 应该在 checkpoint 不存在时抛出 FileNotFoundError")


def main() -> None:
    test_discretize_mouse_nearest_bin()
    test_discretize_mouse_clip_bound()
    test_circular_minimap_mask()
    test_load_model_checkpoint_not_found()
    print("[TEST] 高优先级修复回归测试通过")


if __name__ == "__main__":
    main()
