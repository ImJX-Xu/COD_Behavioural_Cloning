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
KEY_PROB_THRESHOLD = 0.1

# 调试开关：在推理时打印鼠标离散档位与实际像素位移（用于确认模型是否输出非零移动）
DEBUG_PRINT_MOUSE_INFER = True

# 图像与模型风格：已废弃，旧灰度堆叠方案已移除，当前始终使用 EfficientNet + ConvLSTM2D。
# 保留此变量仅为兼容推理脚本中的遗留分支判断，勿修改为 False。
USE_EFFICIENTNET = True  # 废弃开关，勿改

# 序列级 LSTM 开关：
# - True: 使用 EfficientNet + ConvLSTM2D + LSTM 的两级时序结构
# - False: 只使用 EfficientNet + ConvLSTM2D（更轻量，适合做 ablation 或数据较少时）
USE_SEQUENCE_LSTM = False

# 拼接后单帧尺寸（高×宽）：主画面 320×90 + 小地图 90×90 → 410×90
IMG_HEIGHT = 90
IMG_WIDTH = 410
# 主画面与小地图 resize 目标（宽×高，供 cv2.resize）
MAIN_TARGET_SIZE = (320, 90)
MINIMAP_TARGET_SIZE = (90, 90)

# 采集/推理帧率（与论文 Example 一致）
# 降低此值可减少推理延迟，但时序信息会变稀疏
TARGET_FPS = 16

# LSTM 序列长度：每个训练样本包含的时间步数
# 16 FPS 下 SEQ_LEN=32 约覆盖 2 秒上下文，是显存与时序信息的折中
# 增大可捕捉更长行为节奏，但显存占用线性增加
SEQ_LEN = 32

# 鼠标离散档位：中间密、两边疏（由 mouse_bins.py 数据驱动生成）
# 修改档位后需重新训练，旧模型输出头维度不匹配
MOUSE_X_POSSIBLES = [
    -216.7, -44.6, -33.5, -28.0, -22.3, -11.2, -9.0, -4.0, -3.0, -1.0, -0.0, 0.0,
    0.0, 1.0, 3.0, 4.0, 9.0, 11.2, 22.3, 28.0, 33.5, 44.6, 216.7,
]
MOUSE_Y_POSSIBLES = [
    -49.5, -9.0, -6.0, -3.0, -2.0, -0.4, -0.0, 0.0, 0.0, 0.4, 2.0, 3.0,
    6.0, 9.0, 49.5,
]
N_MOUSE_X = len(MOUSE_X_POSSIBLES)
N_MOUSE_Y = len(MOUSE_Y_POSSIBLES)

# 轨迹切块参数：将长轨迹切成 20–30 秒片段再滑窗采样
# 16 FPS 下：CHUNK_MIN_FRAMES=320 ≈ 20s，CHUNK_MAX_FRAMES=480 ≈ 30s
# 切块降低长轨迹偏置，使 train/val 切分更均衡
CHUNK_MIN_FRAMES = 320
CHUNK_MAX_FRAMES = 480

# 训练相关超参数（统一在此处集中管理）
BATCH_SIZE = 8            # 每次前向/反向的样本数；受显存限制，时序模型建议 4~8
ACCUM_STEPS = 1           # 梯度累积步数：每 ACCUM_STEPS 个 batch 更新一次参数
                          # 等效 batch = BATCH_SIZE * ACCUM_STEPS；可在显存不足时模拟大 batch
LR = 3e-4                 # 初始学习率；AdamW 下 3e-4 是经验默认值
WEIGHT_DECAY = 1e-5       # L2 正则化强度；抑制过拟合，数据量小时适当增大
LR_STEP_SIZE = 5          # 学习率衰减间隔（单位：epoch）
                          # 每隔此值触发一次 lr *= LR_GAMMA
                          # 必须 < MAX_EPOCHS 才能生效；原值 100 在 MAX_EPOCHS=10 下永不触发
LR_GAMMA = 0.5            # 学习率衰减倍率：每次触发后 lr 乘以此值
                          # 0.1 衰减过激（缩小 10 倍），0.5 更温和，适合小 epoch 数训练
EARLY_STOP_PATIENCE = 10  # 验证集连续此轮数无提升则提前停止，防过拟合并节省算力
MAX_EPOCHS = 10           # 训练 epoch 上限；CLI/菜单传入的 num_epochs 会被此值截断

# 推理相关：鼠标位移平滑策略
# IS_SPLIT_MOUSE=True：将位移拆为两次各 dx/2 的 moveRel，中间间隔 15ms
# 使视角转动更顺滑，避免单次大跳造成的突兀感（与 Example 一致）
IS_SPLIT_MOUSE = True

# 损失权重：控制各动作头在总损失中的占比
# 启用键盘时使用 WITH_KEY_* 权重，否则使用 NO_KEY_* 权重
# 鼠标 X 和 Y 共享同一系数，各占该系数的一半
# 注意：NO_KEY 两项之和、WITH_KEY 三项之和应均为 1.0，否则损失尺度会漂移
LOSS_WEIGHT_MOUSE_NO_KEY   = 0.5   # 无键盘时：鼠标 X+Y 合计权重（X 和 Y 各占 0.25）
LOSS_WEIGHT_BTN_NO_KEY     = 0.5   # 无键盘时：鼠标左右键权重
LOSS_WEIGHT_MOUSE_WITH_KEY = 0.35  # 有键盘时：鼠标 X+Y 合计权重（X 和 Y 各占 0.175）
LOSS_WEIGHT_BTN_WITH_KEY   = 0.30  # 有键盘时：鼠标左右键权重
LOSS_WEIGHT_KEYS_WITH_KEY  = 0.30  # 有键盘时：键盘多标签权重
