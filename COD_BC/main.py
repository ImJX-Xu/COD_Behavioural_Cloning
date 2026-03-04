import argparse
import os

# 功能开关总控（定义在 config.py，此处引用以便启动时展示）
# 修改 config.py 中的 USE_MINIMAP / USE_KEYBOARD 即可全局生效
from config import USE_KEYBOARD, USE_MINIMAP


def main():
    """
    项目统一入口：
    - 支持命令行子命令：collect/train/infer/eval/clear-data
    - 如果用户直接运行 `python main.py`，则进入交互式菜单模式，
      通过数字选择要执行的流程（更适合日常使用）。

    之所以把真正的业务逻辑放在各个模块里（data_collector/trainer/...），
    而这里只负责参数解析和路由，是为了：
      1. 避免在没有 GUI 环境时就加载截图/键鼠库导致崩溃；
      2. 让采集、训练、推理可以被其它脚本重用（例如未来做批量训练）。
    """
    # 顶层 ArgumentParser，只解析「要做什么」（command）以及通用选项说明。
    # 具体的子功能（采集/训练/推理/评估）都挂在 subparsers 上。
    parser = argparse.ArgumentParser(description="COD LSTM-BC Agent")
    # subparsers 允许我们写出类似 `python main.py train --data ...` 这样的子命令风格接口。
    # dest="command" 表示解析后会在 args.command 里保存子命令名称（如 "train"）。
    subparsers = parser.add_subparsers(dest="command")

    # 子命令：collect —— 专家数据采集
    # 使用示例：python main.py collect --output data/xxx.h5
    collect_parser = subparsers.add_parser("collect", help="Collect expert trajectories")
    collect_parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("data", "expert.h5"),
        help=(
            "采集到的专家轨迹保存位置，HDF5 格式。"
            "如果文件已存在，将在其中追加 traj_x 分组。"
        ),
    )

    # 子命令：train —— 训练 LSTM-BC 模型
    # 使用示例：python main.py train --data data/expert.h5 --epochs 200
    train_parser = subparsers.add_parser("train", help="Train LSTM-BC model")
    train_parser.add_argument(
        "--data",
        type=str,
        default=os.path.join("data", "expert.h5"),
        help="训练数据集路径（由 collect 生成的 h5 文件）。",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="训练轮数上限。早停逻辑会在验证集长期不提升时提前终止。",
    )

    # 子命令：infer —— 在 COD 中实时推理并控制键鼠
    # 使用示例：python main.py infer --checkpoint models/best_model.pt
    infer_parser = subparsers.add_parser("infer", help="Run real-time inference in COD")
    infer_parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join("models", "best_model.pt"),
        help=(
            "已经训练好的模型权重路径。"
            "默认使用训练阶段保存的最优 checkpoint。"
        ),
    )

    # 子命令：eval —— 离线评估模型性能（动作准确率、延迟等）
    # 当前 eval 逻辑是占位实现，后续可以在 evaluator.py 中扩展。
    eval_parser = subparsers.add_parser("eval", help="Evaluate trained agent")
    eval_parser.add_argument(
        "--data",
        type=str,
        default=os.path.join("data", "expert.h5"),
        help="用于评估的轨迹数据路径（通常与训练数据相同或其子集）。",
    )
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join("models", "best_model.pt"),
        help="待评估的模型 checkpoint 路径。",
    )

    # 子命令：data-info —— 查看采集数据详情（不加载模型）
    info_parser = subparsers.add_parser("data-info", help="View collected dataset details")
    info_parser.add_argument(
        "--data",
        type=str,
        default=os.path.join("data", "expert.h5"),
        help="要查看的 h5 数据文件路径。",
    )

    args = parser.parse_args()

    # 交互式菜单：如果没有传子命令（直接 `python main.py`），
    # 就在终端里列出常用功能，用户输入数字来选择。
    # 这种方式比记住一长串命令行参数要友好很多。
    if args.command is None:
        print("=== COD LSTM-BC 智能体 ===")
        print(f"当前配置: 小地图={USE_MINIMAP}, 键盘={USE_KEYBOARD} (修改 config.py 可更改)")
        print("请选择要执行的操作：")
        print("  1) 数据采集 (collect)")
        print("  2) 模型训练 (train)")
        print("  3) 实时推理 (infer)")
        print("  4) 评估 (eval)")
        print("  5) 删除所有采集数据 (clear-data)")
        print("  6) 查看采集数据详情 (data-info)")
        choice = input("请输入数字并回车：").strip()
        if choice == "1":
            args.command = "collect"
            args.output = os.path.join("data", "expert.h5")
        elif choice == "2":
            args.command = "train"
            args.data = os.path.join("data", "expert.h5")
            args.epochs = 200
        elif choice == "3":
            args.command = "infer"
            args.checkpoint = os.path.join("models", "best_model.pt")
        elif choice == "4":
            args.command = "eval"
            args.data = os.path.join("data", "expert.h5")
            args.checkpoint = os.path.join("models", "best_model.pt")
        elif choice == "5":
            args.command = "clear-data"
        elif choice == "6":
            args.command = "data-info"
            args.data = os.path.join("data", "expert.h5")
        else:
            print("无效选择，程序退出。")
            return

    # 为了避免在没有图形界面的环境里导入 mss/pyautogui 等库导致错误，
    # 这里采用「惰性导入」：根据最终选择的 command 再去 import 对应模块。
    if args.command == "collect":
        from src.data_collector import run_data_collection
        run_data_collection(output_path=args.output)
    elif args.command == "train":
        from src.trainer import run_training
        run_training(data_path=args.data, num_epochs=args.epochs)
    elif args.command == "infer":
        from src.inferencer import run_inference
        run_inference(checkpoint_path=args.checkpoint)
    elif args.command == "eval":
        from src.evaluator import run_evaluation
        run_evaluation(data_path=args.data, checkpoint_path=args.checkpoint)
    elif args.command == "data-info":
        from src.data_info import run_data_info
        run_data_info(data_path=args.data)
    elif args.command == "clear-data":
        # 安全删除 data 下的采集数据：只删除普通文件，不动目录本身，
        # 并且要求用户在终端里再次确认，避免误删长时间采集的轨迹。
        data_dir = "data"
        if not os.path.isdir(data_dir):
            print("data 目录不存在，无需删除。")
            return
        confirm = input("确认删除 data 目录下所有采集数据文件？(y/N): ").strip().lower()
        if confirm != "y":
            print("已取消删除。")
            return
        removed = 0
        for name in os.listdir(data_dir):
            path = os.path.join(data_dir, name)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                    removed += 1
                except OSError:
                    pass
        print(f"已删除 data 目录下 {removed} 个文件。")


if __name__ == "__main__":
    main()

