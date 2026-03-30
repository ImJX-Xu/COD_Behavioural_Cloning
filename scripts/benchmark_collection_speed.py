"""
采集速度基准测试：测量当前硬件能否满足 TARGET_FPS 的采集需求。
运行约 5 秒，输出每帧耗时、实际 FPS、是否满足目标。
"""
import os
import sys
import time

import cv2
import mss
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TARGET_FPS, USE_EFFICIENTNET, MAIN_TARGET_SIZE, MINIMAP_TARGET_SIZE

# 与 data_collector 一致
from src.data_collector import (
    crop_main_center_half,
    MINIMAP_CENTER_REL,
    _minimap_center_to_roi_abs,
    preprocess_frame,
    preprocess_frame_roi,
    apply_circular_minimap_mask,
)

TARGET_INTERVAL = 1.0 / TARGET_FPS
DURATION_SEC = 5.0


def preprocess_one_frame(img: np.ndarray, monitor: dict) -> np.ndarray:
    """与采集管线一致：主 (320,90) + 小地图 (90,90) 拼接 (90,410,3)，或灰度单图。"""
    if USE_EFFICIENTNET:
        img_main = crop_main_center_half(img, monitor)
        frame_main = preprocess_frame(img_main, MAIN_TARGET_SIZE)
        roi = _minimap_center_to_roi_abs(monitor, MINIMAP_CENTER_REL)
        frame_mm = preprocess_frame_roi(img, roi, MINIMAP_TARGET_SIZE)
        apply_circular_minimap_mask(frame_mm)
        return np.hstack([frame_main, frame_mm])
    img_main = crop_main_center_half(img, monitor)
    return preprocess_frame(img_main)


def main():
    print("=" * 50)
    print("COD_BC 采集速度基准测试")
    print("=" * 50)
    print(f"目标帧率: {TARGET_FPS} FPS (每帧 ≤ {TARGET_INTERVAL*1000:.1f} ms)")
    print(f"测试时长: {DURATION_SEC} 秒")
    print()

    with mss.mss() as sct:
        mon = sct.monitors[1]
        monitor = {"left": mon["left"], "top": mon["top"], "width": mon["width"], "height": mon["height"]}
        w, h = monitor["width"], monitor["height"]
        print(f"捕获区域: {w}x{h}")

        frame_times = []
        start = time.perf_counter()
        n = 0

        while (time.perf_counter() - start) < DURATION_SEC:
            t0 = time.perf_counter()
            raw = sct.grab(monitor)
            img = np.array(raw, dtype=np.uint8)
            frame = preprocess_one_frame(img, monitor)
            t1 = time.perf_counter()
            frame_times.append((t1 - t0) * 1000)
            n += 1

        elapsed = time.perf_counter() - start

    if not frame_times:
        print("未采集到任何帧")
        return

    frame_times = np.array(frame_times)
    actual_fps = n / elapsed
    avg_ms = frame_times.mean()
    max_ms = frame_times.max()
    min_ms = frame_times.min()
    p99_ms = np.percentile(frame_times, 99)

    print()
    print("结果:")
    print(f"  采集帧数:     {n}")
    print(f"  实际耗时:     {elapsed:.2f} 秒")
    print(f"  实际 FPS:     {actual_fps:.1f}")
    print(f"  每帧耗时:     平均 {avg_ms:.1f} ms, 最小 {min_ms:.1f} ms, 最大 {max_ms:.1f} ms, P99 {p99_ms:.1f} ms")
    print()

    if avg_ms <= TARGET_INTERVAL * 1000:
        print(f"  [OK] 平均耗时 {avg_ms:.1f} ms <= 目标 {TARGET_INTERVAL*1000:.1f} ms，硬件可满足 {TARGET_FPS} FPS")
    else:
        print(f"  [!!] 平均耗时 {avg_ms:.1f} ms > 目标 {TARGET_INTERVAL*1000:.1f} ms，可能无法稳定维持 {TARGET_FPS} FPS")
        print(f"       建议: 降低 TARGET_FPS 或减小分辨率")

    if p99_ms > TARGET_INTERVAL * 1000 * 1.5:
        print(f"  [!] P99 耗时 {p99_ms:.1f} ms 较高，偶发卡顿可能影响采集质量")

    print("=" * 50)


if __name__ == "__main__":
    main()
