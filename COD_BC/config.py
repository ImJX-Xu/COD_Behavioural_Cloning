"""
功能开关总控：所有模块从此处读取，修改此处即可全局生效。
"""
from math import inf

# 小地图（左上角 ROI）：采集/训练/推理是否使用
USE_MINIMAP = False

# 键盘动作是否参与整个管线（采集/训练/推理）
USE_KEYBOARD = False

# 分键位控制：
# - TRAIN_KEYS_WASD: 是否在训练中对 W/A/S/D 四个移动键计算 BCE 损失
# - TRAIN_KEYS_SHIFT3: 是否在训练中对 Shift/3 计算 BCE 损失
TRAIN_KEYS_WASD = False
TRAIN_KEYS_SHIFT3 = False

# 推理时键盘触发阈值（sigmoid 概率 > THRESHOLD 视为按下）
KEY_PROB_THRESHOLD = 0.3

# 调试开关：在推理时打印鼠标离散档位与实际像素位移（用于确认模型是否输出非零移动）
DEBUG_PRINT_MOUSE_INFER = True

# 图像与模型风格：当前仅保留彩色 EfficientNet + ConvLSTM2D（旧灰度堆叠方案已移除）
USE_EFFICIENTNET = True

# 拼接后单帧尺寸（高×宽）：主画面 320×90 + 小地图 90×90 → 410×90
IMG_HEIGHT = 90
IMG_WIDTH = 410
# 主画面与小地图 resize 目标（宽×高，供 cv2.resize）
MAIN_TARGET_SIZE = (320, 90)
MINIMAP_TARGET_SIZE = (90, 90)

# 与 Example 一致：采集/推理帧率、LSTM 序列长度
TARGET_FPS = 16  # loop_fps=16（Example）
SEQ_LEN = 48     # 序列长度（原 Example 为 96；改为 48 以降低显存）

# 鼠标离散档位：与 Example (Counter-Strike_Behavioural_Cloning) 完全一致，中间密、两边疏
MOUSE_X_POSSIBLES = [
    -1000.0, -500.0, -300.0, -200.0, -100.0, -60.0, -30.0, -20.0, -10.0,
    -4.0, -2.0, -0.0, 2.0, 4.0, 10.0, 20.0, 30.0, 60.0, 100.0,
    200.0, 300.0, 500.0, 1000.0,
]
MOUSE_Y_POSSIBLES = [
    -200.0, -100.0, -50.0, -20.0, -10.0, -4.0, -2.0, -0.0,
    2.0, 4.0, 10.0, 20.0, 50.0, 100.0, 200.0,
]
N_MOUSE_X = len(MOUSE_X_POSSIBLES)
N_MOUSE_Y = len(MOUSE_Y_POSSIBLES)

# 训练相关超参数（统一在此处集中管理）
BATCH_SIZE = 4
ACCUM_STEPS = 1          # 每 ACCUM_STEPS 个 batch 更新一次，等效 batch = BATCH_SIZE * ACCUM_STEPS
LR = 3e-4
WEIGHT_DECAY = 1e-5
LR_STEP_SIZE = 100        # 学习率衰减 step（epoch）
LR_GAMMA = 0.1            # 学习率衰减倍率
EARLY_STOP_PATIENCE = 10  # 验证集无提升多少个 epoch 后早停
MAX_EPOCHS = 50          # 训练 epoch 上限；CLI/菜单传入的 num_epochs 会被此值截断

# 推理相关：是否做 IS_SPLIT_MOUSE 式平滑：位移减半、同帧内应用两次，观感更顺（与 Example 一致）
IS_SPLIT_MOUSE = True
