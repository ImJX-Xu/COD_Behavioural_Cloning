import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


def load_mouse_arrays(h5_path: str):
    import h5py
    import numpy as np

    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)

    dx_all = []
    dy_all = []
    with h5py.File(h5_path, "r") as f:
        for name in sorted(f.keys()):
            if not name.startswith("traj_"):
                continue
            grp = f[name]
            if "mouse_dx" not in grp or "mouse_dy" not in grp:
                continue
            dx = np.asarray(grp["mouse_dx"][:], dtype="float32")
            dy = np.asarray(grp["mouse_dy"][:], dtype="float32")
            if dx.size == 0 or dy.size == 0:
                continue
            dx_all.append(dx)
            dy_all.append(dy)

    if not dx_all:
        raise RuntimeError("没有在任何 traj_* 里找到 mouse_dx/mouse_dy")

    dx_all = np.concatenate(dx_all, axis=0)
    dy_all = np.concatenate(dy_all, axis=0)
    return dx_all, dy_all


def print_stats(dx, dy):
    import numpy as np

    print("========== Mouse stats ==========")
    for name, arr in [("dx", dx), ("dy", dy)]:
        abs_a = np.abs(arr)
        print(f"{name}: count={arr.size}")
        print(f"  mean={arr.mean():.3f}, std={arr.std():.3f}, min={arr.min():.1f}, max={arr.max():.1f}")
        print(
            f"  |{name}| percentiles: "
            f"50%={np.percentile(abs_a,50):.2f}, "
            f"75%={np.percentile(abs_a,75):.2f}, "
            f"90%={np.percentile(abs_a,90):.2f}, "
            f"95%={np.percentile(abs_a,95):.2f}, "
            f"99%={np.percentile(abs_a,99):.2f}"
        )
    print("=================================")


def build_symmetric_bins(values: Any, n_bins: int):
    """
    根据绝对值分布，用分位数在 [0, p_max] 上均匀取点，
    构造对称挡位列表，如 [-a_k, ..., -a_1, 0, a_1, ..., a_k]。
    """
    import numpy as np

    if n_bins % 2 == 0:
        raise ValueError("n_bins 必须是奇数，方便对称 + 0 档")

    abs_vals = np.abs(values)
    if abs_vals.size == 0:
        return [0.0] * n_bins

    # 录制数据真实上下限（绝对值最大），用于保证挡位有「冗余」空间
    data_max_abs = float(abs_vals.max())
    if data_max_abs <= 0:
        return [0.0] * n_bins

    # 为了鲁棒性：在真实范围基础上预留一定冗余（例如 10%）
    margin_factor = 1.10
    target_max_abs = data_max_abs * margin_factor

    # p_max 只用于构造「中间」档位的分布，不再决定总范围上限
    p_max = 99.5  # 避免极少数异常点把中间档拉得太大
    max_abs_p = float(np.percentile(abs_vals, p_max))
    if max_abs_p <= 0:
        max_abs_p = data_max_abs

    half = (n_bins - 1) // 2
    # 在线性分位数上取 half 个绝对值挡位（用于中间到中大范围）
    qs = np.linspace(0.0, p_max, num=half + 1)[1:]  # 去掉 0%，避免重复 0 档
    levels = np.percentile(abs_vals, qs)
    # 去重 + 排序（限制在基于分位数的 max_abs_p 范围内）
    levels = np.unique(np.clip(levels, 0.0, max_abs_p))

    # 如果 unique 后档位不足 half 个，就简单插值补齐（仍限制在 max_abs_p 内）
    if len(levels) < half:
        extra = np.linspace(0.0, max_abs_p, num=half + 1)[1:]
        levels = np.unique(np.concatenate([levels, extra]))
    # 只保留最靠近的 half 个
    levels = np.sort(levels)[:half]

    # 确保最大挡位略大于录制数据的最大绝对位移，保证一定鲁棒性
    if levels.size == 0:
        # 极端情况下全部为 0，这里人为构造一个线性档位
        levels = np.linspace(data_max_abs * 0.1, target_max_abs, num=half)
    else:
        if levels[-1] < target_max_abs:
            levels[-1] = target_max_abs

    neg = -levels[::-1]
    pos = levels
    bins = list(neg) + [0.0] + list(pos)
    return [float(x) for x in bins]


def _format_bins_list(values: Sequence[float], *, items_per_line: int = 12) -> str:
    if items_per_line <= 0:
        raise ValueError("items_per_line 必须为正整数")
    chunks = [values[i : i + items_per_line] for i in range(0, len(values), items_per_line)]
    lines = []
    for chunk in chunks:
        lines.append("    " + ", ".join(f"{v:.1f}" for v in chunk) + ",")
    return "[\n" + "\n".join(lines) + "\n]"


def _replace_list_assignment(src: str, var_name: str, new_list_literal: str) -> str:
    """
    替换形如：
        VAR = [
            ...
        ]
    的整段内容（允许跨行，且只替换第一次出现）。
    """
    pattern = re.compile(rf"(?ms)^{re.escape(var_name)}\s*=\s*\[.*?\]\s*")
    m = pattern.search(src)
    if not m:
        raise RuntimeError(f"在 config 中未找到 `{var_name} = [...]`，无法自动写入")

    replacement = f"{var_name} = {new_list_literal}\n"
    return src[: m.start()] + replacement + src[m.end() :]


def apply_bins_to_config(
    config_path: str,
    *,
    x_bins,
    y_bins,
    backup: bool = True,
    items_per_line: int = 12,
) -> str:
    cfg = Path(config_path)
    if not cfg.exists():
        raise FileNotFoundError(str(cfg))

    text = cfg.read_text(encoding="utf-8")
    new_x = _format_bins_list(x_bins, items_per_line=items_per_line)
    new_y = _format_bins_list(y_bins, items_per_line=items_per_line)

    updated = _replace_list_assignment(text, "MOUSE_X_POSSIBLES", new_x)
    updated = _replace_list_assignment(updated, "MOUSE_Y_POSSIBLES", new_y)

    if backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = cfg.with_name(f"{cfg.stem}.bak_{stamp}{cfg.suffix}")
        bak.write_text(text, encoding="utf-8")

    cfg.write_text(updated, encoding="utf-8")
    return str(cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/expert.h5")
    parser.add_argument("--nx", type=int, default=23, help="Number of X-axis bins (must be odd)")
    parser.add_argument("--ny", type=int, default=15, help="Number of Y-axis bins (must be odd)")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "COD_BC" / "config.py"),
        help="Path to COD_BC/config.py (default: ./COD_BC/config.py)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write bins into config.py (replaces MOUSE_X/Y_POSSIBLES; makes a backup by default)",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Do not prompt to apply (useful for CI/non-interactive runs)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="With --apply: do not create a backup file (not recommended)",
    )
    parser.add_argument(
        "--items-per-line",
        type=int,
        default=12,
        help="When writing config.py, how many values per line",
    )
    args = parser.parse_args()

    dx, dy = load_mouse_arrays(args.data)
    print_stats(dx, dy)

    print(f"\n根据当前数据分布，推荐的挡位（X 轴，共 {args.nx} 档）：")
    x_bins = build_symmetric_bins(dx, args.nx)
    print("MOUSE_X_POSSIBLES = [")
    for v in x_bins:
        print(f"    {v:.1f},")
    print("]")

    print(f"\n推荐的挡位（Y 轴，共 {args.ny} 档）：")
    y_bins = build_symmetric_bins(dy, args.ny)
    print("MOUSE_Y_POSSIBLES = [")
    for v in y_bins:
        print(f"    {v:.1f},")
    print("]")

    if args.apply:
        apply_bins_to_config(
            args.config,
            x_bins=x_bins,
            y_bins=y_bins,
            backup=(not args.no_backup),
            items_per_line=args.items_per_line,
        )
        print(f"\n✅ 已写入：{args.config}")
        if not args.no_backup:
            print("（已在同目录生成 config 备份：config.bak_YYYYmmdd_HHMMSS.py）")
    elif (not args.no_prompt) and sys.stdin.isatty():
        resp = input(f"\n是否将以上挡位写入到 `{args.config}` ？输入 y 确认（默认不写入）: ").strip().lower()
        if resp in {"y", "yes"}:
            apply_bins_to_config(
                args.config,
                x_bins=x_bins,
                y_bins=y_bins,
                backup=(not args.no_backup),
                items_per_line=args.items_per_line,
            )
            print(f"\n✅ 已写入：{args.config}")
            if not args.no_backup:
                print("（已在同目录生成 config 备份：config.bak_YYYYmmdd_HHMMSS.py）")
        else:
            print("\n未写入（如需写入，可重新运行并输入 y，或使用 --apply）。")


if __name__ == "__main__":
    main()