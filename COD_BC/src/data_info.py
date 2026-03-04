"""
查看当前采集数据的详细信息，不加载模型，仅读取 h5 元数据。
"""

import os
from typing import Optional


def run_data_info(data_path: str) -> None:
    """
    打印 h5 采集文件的详细信息：
    - 轨迹数量、各轨迹帧数、总帧数
    - 分辨率、采样帧率
    - 可训练序列数估算
    """
    if not os.path.exists(data_path):
        print(f"[DATA-INFO] 文件不存在: {data_path}")
        return

    try:
        import h5py
    except ImportError:
        print("[DATA-INFO] 需要 h5py 库")
        return

    file_size_mb = os.path.getsize(data_path) / (1024 * 1024)

    with h5py.File(data_path, "r") as f:
        traj_names = sorted([n for n in f.keys() if n.startswith("traj_")])
        if not traj_names:
            print(f"[DATA-INFO] 文件中无 traj_* 轨迹组，顶层键: {list(f.keys())}")
            return

        total_frames = 0
        details: list[str] = []

        # 鼠标采样统计：全局绝对值均值/最大值，用于快速检查是否真正录到了鼠标移动
        has_mouse_stats = False
        mouse_dx_max = 0.0
        mouse_dy_max = 0.0
        mouse_dx_sum_abs = 0.0
        mouse_dy_sum_abs = 0.0
        mouse_count = 0

        for name in traj_names:
            grp = f[name]
            if "states" not in grp:
                details.append(f"    {name}: (无 states 数据集，可能格式不兼容)")
                continue
            st = grp["states"]
            n_frames = st.shape[0]
            total_frames += n_frames
            h, w = st.shape[1], st.shape[2]
            duration_sec = n_frames / grp.attrs.get("target_fps", 30)
            details.append(f"    {name}: {n_frames} 帧 ({duration_sec:.1f}s), 分辨率 {w}x{h}")

            if "mouse_dx" in grp and "mouse_dy" in grp:
                import numpy as np

                dx = np.asarray(grp["mouse_dx"][:], dtype="float32")
                dy = np.asarray(grp["mouse_dy"][:], dtype="float32")
                if dx.size and dy.size:
                    has_mouse_stats = True
                    mouse_dx_max = max(mouse_dx_max, float(np.max(np.abs(dx))))
                    mouse_dy_max = max(mouse_dy_max, float(np.max(np.abs(dy))))
                    mouse_dx_sum_abs += float(np.sum(np.abs(dx)))
                    mouse_dy_sum_abs += float(np.sum(np.abs(dy)))
                    mouse_count += dx.size

        # 公共元数据（取第一条有 states 的轨迹）
        grp0 = None
        for n in traj_names:
            if "states" in f[n]:
                grp0 = f[n]
                break
        if grp0 is None:
            print("[DATA-INFO] 所有轨迹组均无 states 数据集，无法解析。")
            return
        monitor_w = grp0.attrs.get("monitor_width", "?")
        monitor_h = grp0.attrs.get("monitor_height", "?")
        target_fps = grp0.attrs.get("target_fps", "?")
        has_minimap = "states_minimap" in grp0
        has_keys = "keys" in grp0

        from config import SEQ_LEN
        # 可训练序列数估算（与 data_processor 一致）
        trainable = sum(max(0, f[n]["states"].shape[0] - SEQ_LEN) for n in traj_names if "states" in f[n])

        st0 = grp0["states"]
        if st0.ndim == 4 and st0.shape[-1] in (3, 4):
            color_mode = "RGB"
        else:
            color_mode = "灰度"
        h, w = st0.shape[1], st0.shape[2]

    print("========== 采集数据详情 ==========")
    print(f"文件: {data_path}")
    print(f"大小: {file_size_mb:.2f} MB")
    print("---------------------------------")
    print(f"轨迹数: {len(traj_names)}")
    print(f"总帧数: {total_frames}")
    print(f"主屏分辨率: {monitor_w}x{monitor_h}")
    print(f"采样帧率: {target_fps} FPS")
    print(f"主界面尺寸: {w}x{h} ({color_mode})")
    print(f"含小地图通道: {has_minimap}")
    print(f"含键盘标签:   {has_keys}")
    if has_mouse_stats and mouse_count > 0:
        avg_dx = mouse_dx_sum_abs / mouse_count
        avg_dy = mouse_dy_sum_abs / mouse_count
        print(f"鼠标采样统计: mean|dx|≈{avg_dx:.2f}, max|dx|≈{mouse_dx_max:.2f}, mean|dy|≈{avg_dy:.2f}, max|dy|≈{mouse_dy_max:.2f}")
    print("---------------------------------")
    print("各轨迹:")
    for line in details:
        print(line)
    print("---------------------------------")
    print(f"可训练序列数(seq_len={SEQ_LEN}): 约 {trainable}")
    print("=================================")
