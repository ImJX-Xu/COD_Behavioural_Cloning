## COD_BC：基于 LSTM 的 COD 行为克隆智能体

**本项目基于 [Counter-Strike_Behavioural_Cloning](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning) 进行二次开发，感谢原作者 Tim Pearce、Jun Zhu 的开源贡献。**

- **原仓库**：[TeaPearce/Counter-Strike_Behavioural_Cloning](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning)（IEEE CoG 2022 论文 *Counter-Strike Deathmatch with Large-Scale Behavioural Cloning* 官方实现）
- **原项目协议**：原项目仅授权个人项目与开源研究使用，**不授予任何形式的商业使用许可**。本衍生项目继承并遵守该约定，详见仓库内 [LICENSE](LICENSE)。
- **主要修改与适配**：
  1. **目标游戏**：由 CS:GO 改为《使命召唤》(Call of Duty)，画面裁切、分辨率与 ROI 按 COD 适配；
  2. **技术栈**：由 TensorFlow/Keras 改为 **PyTorch**，模型为 EfficientNetB0 + ConvLSTM2D + LSTM；
  3. **数据管线**：采集使用 `mss` + `pynput`，存储为 HDF5，序列长度与离散化方案独立配置；
  4. **控制与推理**：实时截屏 + 前向推理 + `pyautogui` 控制键鼠，支持 F9/F12 启停与调试输出；
  5. **工具与配置**：统一 `config.py`、`mouse_bins.py` 挡位推荐、数据统计与评估脚本等。

---

本项目实现了一个用于《使命召唤》(Call of Duty) 的**行为克隆 (Behavioural Cloning)** 智能体，整体流程参考了论文与官方仓库 **Counter-Strike_Behavioural_Cloning (Example)**，并针对 COD 和 Windows 环境做了全面适配。

完整链路：

- **画面采集**：从主显示器抓取 COD 全屏画面，裁主画面 + 小地图，生成 `(90, 410, 3)` RGB 帧。
- **轨迹与动作记录**：每帧记录鼠标位移、左右键、键盘按键集合。
- **预处理与序列构建**：切割轨迹 → 固定长度序列 (`SEQ_LEN=48`)，离散化鼠标 → 分类标签。
- **模型**：EfficientNetB0 (中间层) + ConvLSTM2D + LSTM，多头输出鼠标 X/Y、左右键、键盘。
- **训练**：小批量 + 梯度累积，支持键盘头可选训练（当前仅鼠标 + 鼠标键）。
- **实时推理**：实时截屏 → 前向推理 → `pyautogui` 发出鼠标/键盘操作，直接控制 COD。
- **调试工具**：数据统计、挡位自动推荐、鼠标采样质量检查、训练/推理调试输出等。

---

## 目录结构概览

只列与 COD_BC 行为克隆相关的主要部分：

```text
COD_BC/
├── data/                  # 采集到的专家轨迹 (expert.h5 等)
├── logs/                  # TensorBoard 日志
├── models/                # 训练得到的模型权重 (best_model.pt 等)
├── src/
│   ├── data_collector.py  # 数据采集（屏幕 + 鼠标 + 键盘）
│   ├── data_processor.py  # HDF5 预处理与 Dataset 构建
│   ├── model.py           # EfficientNet + ConvLSTM + LSTM 模型
│   ├── trainer.py         # 训练循环与日志
│   ├── inferencer.py      # 实时推理并控制 COD
│   ├── evaluator.py       # 离线评估工具
│   └── data_info.py       # 数据详情与鼠标采样统计
├── docs/
│   ├── 训练流程.md                      # 完整流程设计说明
│   └── 项目全链路技术说明_面试版.md      # 面试技术说明
├── mouse_bins.py          # 鼠标档位数据驱动推荐工具
├── main.py                # CLI + 交互式菜单入口
└── requirements.txt       # 依赖列表
```

仓库根目录结构：

```text
BC_COD/
├── COD_BC/        # 主项目包（见上）
├── docs/          # 仓库级开发笔记
├── references/    # 参考论文 PDF
├── README.md
├── Dockerfile
└── .gitignore
```

---

## 环境准备

推荐使用 Conda：

```bash
conda create -n COD_BC python=3.10
conda activate COD_BC

pip install -r COD_BC/requirements.txt
```

确认 CUDA 可用（可选）：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 配置总控：`COD_BC/config.py`

所有重要开关和训练/推理相关超参数都集中在这里。

### 功能开关

```python
# 小地图（左上角 ROI）：采集/训练/推理是否使用
USE_MINIMAP = False

# 键盘动作是否参与整个管线（采集/训练/推理）
USE_KEYBOARD = False

# 分键位控制（仅影响训练 loss）
TRAIN_KEYS_WASD   = False  # W/A/S/D 四个键是否参与键盘 BCE 损失
TRAIN_KEYS_SHIFT3 = False  # Shift/3 是否参与键盘 BCE 损失
```

### 键盘推理阈值与调试

```python
KEY_PROB_THRESHOLD = 0.3      # 推理时键盘概率 > 阈值视为按下
DEBUG_PRINT_MOUSE_INFER = False  # True 时在推理里打印鼠标档位和像素位移
```

### 画面与序列

```python
USE_EFFICIENTNET = True

IMG_HEIGHT = 90
IMG_WIDTH  = 410                   # 主画面 320x90 + 小地图 90x90 → 拼接宽 410

MAIN_TARGET_SIZE    = (320, 90)    # 主画面 resize 目标（宽 x 高）
MINIMAP_TARGET_SIZE = (90, 90)     # 小地图 resize 目标（宽 x 高）

TARGET_FPS = 16       # 采集与推理帧率
SEQ_LEN    = 48       # LSTM 序列长度（Example 原始为 96）
```

### 鼠标离散挡位

```python
MOUSE_X_POSSIBLES = [...]  # 通过 mouse_bins.py 自动推荐后填入
MOUSE_Y_POSSIBLES = [...]
N_MOUSE_X = len(MOUSE_X_POSSIBLES)
N_MOUSE_Y = len(MOUSE_Y_POSSIBLES)
```

可以用 `mouse_bins.py` 自动根据已有 `expert.h5` 生成适合当前数据分布的挡位。

### 训练超参数

```python
BATCH_SIZE          = 4
ACCUM_STEPS         = 1          # 每 ACCUM_STEPS 个 batch 更新一次，等效 batch = BATCH_SIZE * ACCUM_STEPS
LR                   = 3e-4
WEIGHT_DECAY         = 1e-5
LR_STEP_SIZE         = 100       # 学习率衰减 step（epoch）
LR_GAMMA             = 0.1       # 学习率衰减倍率
EARLY_STOP_PATIENCE  = 10        # 验证集无提升多少个 epoch 后早停
MAX_EPOCHS           = 50        # 训练 epoch 上限（CLI/菜单传入会被截断）
```

### 推理

```python
IS_SPLIT_MOUSE = True  # 鼠标位移拆成两半分两次 moveRel，观感更顺
```

---

## 统一入口：`COD_BC/main.py`

支持命令行和交互式菜单两种方式。

### 命令行子命令

- **数据采集**：

  ```bash
  python COD_BC/main.py collect --output data/expert.h5
  ```

- **训练**：

  ```bash
  python COD_BC/main.py train --data data/expert.h5 --epochs 200
  ```

  实际训练 epoch 会被 `config.MAX_EPOCHS` 截断。

- **实时推理**：

  ```bash
  python COD_BC/main.py infer --checkpoint models/best_model.pt
  ```

- **离线评估**：

  ```bash
  python COD_BC/main.py eval --data data/expert.h5 --checkpoint models/best_model.pt
  ```

- **查看数据详情**：

  ```bash
  python COD_BC/main.py data-info --data data/expert.h5
  ```

### 交互式菜单

直接运行：

```bash
python COD_BC/main.py
```

会看到：

```text
=== COD LSTM-BC 智能体 ===
当前配置: 小地图=False, 键盘=False (修改 config.py 可更改)
  1) 数据采集 (collect)
  2) 模型训练 (train)
  3) 实时推理 (infer)
  4) 评估 (eval)
  5) 删除所有采集数据 (clear-data)
  6) 查看采集数据详情 (data-info)
```

按数字选择即可。

---

## 数据采集：`src/data_collector.py`

### 截屏与裁切

- 使用 `mss` 从主显示器抓全屏。
- 主画面：取屏幕中心的 1/2 区域。
- 小地图：用 `MINIMAP_CENTER_REL` 在屏幕左上定位方形 ROI，再做圆形遮罩。
- Resize 并拼接为单帧 `(90, 410, 3)`。

### 动作记录

- 鼠标位移：当前帧与上一帧鼠标坐标的差值，单位为像素（连续量）。
- 鼠标左右键：是否按下。
- 键盘：使用 `pynput.keyboard` 监听所有按键，记录当帧的按键集合。
- 时间戳：系统时间秒。

### 写入 HDF5

每次录制一条轨迹 `traj_k`，包含：

- `states`: `[N, 90, 410, 3]` RGB 帧
- `mouse_dx`, `mouse_dy`: `float32[N]`
- `mouse_left`, `mouse_right`: `int8[N]`
- `timestamps`: `float64[N]`
- `keys`: 字符串二维数组（每帧的按键列表，空字符串补齐）
- attrs：`monitor_width/height`、`target_fps` 等

---

## 数据预处理与 Dataset：`src/data_processor.py`

### 轨迹切割：`build_chunked_h5`

- 将长轨迹按 20–30 秒（约 320–480 帧）切分为多个 `chunk_k`。
- 每个 chunk 拷贝对应帧段的 `states/mouse_dx/dy/...`。
- 输出到 `<expert>_chunked.h5`，不修改原始文件。

### 序列构建：`H5SequenceDataset`

对每个 `chunk`：

- 使用滑动窗口采样固定长度序列 `[start, start+SEQ_LEN)`。
- `obs`：将 `states[s:e]` 归一化到 `[0,1]`，并转为 `[T, 3, 90, 410]`。
- 鼠标离散化：

  ```python
  ix, iy = discretize_mouse(mouse_dx, mouse_dy)  # 最近档位
  mouse_x_class[t] = ix
  mouse_y_class[t] = iy
  ```

- 鼠标左右键：`mouse_buttons[t] = [left, right]`。
- 键盘：

  ```python
  ACTION_KEYS = ("w", "a", "s", "d", "shift", "3")
  keys_vec[t] = encode_keyboard_multi_hot(keys_at_t)
  ```

Dataset 返回：

- `obs`: `[T, C, H, W]`
- `mouse_x_class`: `[T]`
- `mouse_y_class`: `[T]`
- `mouse_buttons`: `[T, 2]`
- `keys`: `[T, 6]`

### 数据划分：`create_splits`

基于 `chunk_*` 列表按比例拆分为 train/val/test，并返回对应的 `H5SequenceDataset`。

---

## 模型结构：`src/model.py`


方案：**EfficientNet + ConvLSTM2D + LSTM**。

- Backbone：截断版 `efficientnet_b0`，输出 192 通道中间特征图；
- ConvLSTM：在特征图上做时序卷积 LSTM；
- AdaptiveAvgPool2d + LSTM：将每帧特征池化到 256 维，再经 2 层 LSTM 做时序建模；
- 多头输出：

  - `mouse_x_logits`: `[B, T, N_MOUSE_X]`
  - `mouse_y_logits`: `[B, T, N_MOUSE_Y]`
  - `mouse_buttons_logits`: `[B, T, 2]`
  - `key_logits`: `[B, T, 6]`

---

## 训练：`src/trainer.py`

### DataLoader 与状态/动作空间打印

```python
batch_size  = BATCH_SIZE
accum_steps = ACCUM_STEPS
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
```

会自动打印：

```text
[TRAIN] State space: obs[T,C,H,W]=(48,3,90,410)
[TRAIN] Action space: mouse_x_classes=..., mouse_y_classes=..., mouse_buttons_shape=(T,2), keys_shape=(T,6), ...
```

### Loss 设计

- 鼠标 X/Y：`CrossEntropyLoss`（未来可以在此处加 class weight，对大位移动作升权）。
- 鼠标左右键：`BCEWithLogitsLoss`。
- 键盘：`BCEWithLogitsLoss(reduction="none")` + 配置驱动的掩码：

  ```python
  key_loss_mask_vals = [
      1 if TRAIN_KEYS_WASD   else 0,  # w
      1 if TRAIN_KEYS_WASD   else 0,  # a
      1 if TRAIN_KEYS_WASD   else 0,  # s
      1 if TRAIN_KEYS_WASD   else 0,  # d
      1 if TRAIN_KEYS_SHIFT3 else 0,  # shift
      1 if TRAIN_KEYS_SHIFT3 else 0,  # 3
  ]
  ```

- 总损失：

  ```python
  loss = 0.35*(loss_mouse_x + loss_mouse_y) + 0.3*mouse_btn_loss + 0.3*keys_loss  # 启用键盘时
  # 或
  loss = 0.5*(loss_mouse_x + loss_mouse_y) + 0.5*mouse_btn_loss                  # 只训鼠标时
  ```

### 梯度累积与优化

- 每个 batch 做 `(loss / ACCUM_STEPS).backward()`；
- 每 `ACCUM_STEPS` 个 batch 更新一次参数。
- 优化器与调度器从 `config.py` 读取。
- 早停基于验证集损失与 `EARLY_STOP_PATIENCE`。
- epoch 上限由 `MAX_EPOCHS` 控制。

---

## 实时推理：`src/inferencer.py`

### 控制方式

- F9：开始/暂停由模型控制键鼠；
- F12：停止并退出推理。

### 推理流程

1. 实时截屏 → 裁切主画面 + 小地图 → `(90,410,3)`。
2. 归一化并转成 `[1,1,3,90,410]` 送入模型。
3. 输出：

   ```python
   mouse_x_logits, mouse_y_logits, mouse_btn_logits, key_logits, lstm_state = model(obs, lstm_state)
   ```

4. 鼠标：

   ```python
   idx_x = argmax(mouse_x_logits)
   idx_y = argmax(mouse_y_logits)
   dx   = MOUSE_X_POSSIBLES[idx_x]
   dy   = MOUSE_Y_POSSIBLES[idx_y]
   ```

   若 `DEBUG_PRINT_MOUSE_INFER=True`，会在终端打印档位与位移，方便调试。

5. 鼠标移动：

   ```python
   if IS_SPLIT_MOUSE:
       moveRel(dx/2, dy/2); sleep(0.015); moveRel(dx/2, dy/2)
   else:
       moveRel(dx, dy)
   ```

6. 键盘（`USE_KEYBOARD=True` 时）：根据 `key_probs > KEY_PROB_THRESHOLD` 决定是否按下/抬起 W/A/S/D/Shift/3。

---

## 数据与模型评估

### 数据详情与鼠标采样质量：`src/data_info.py`

```bash
python COD_BC/main.py data-info --data data/expert.h5
```

可看到：

```text
鼠标采样统计: mean|dx|≈..., max|dx|≈..., mean|dy|≈..., max|dy|≈...
```

用于快速判断录制数据中鼠标位移的大致范围。

### 模型评估：`src/evaluator.py`

```bash
python COD_BC/main.py eval --data data/expert.h5 --checkpoint models/best_model.pt
```

会输出：

```text
鼠标 X 分类准确率: xx.xx%
鼠标 Y 分类准确率: yy.yy%
鼠标左右键准确率: zz.zz%
键盘多标签准确率(WASD+Shift+3): ...
完整动作匹配率:                 ...
```

用于评估模型在离线数据上的模仿质量。

---

## 鼠标挡位自动推荐：`mouse_bins.py`

`mouse_bins.py` 会从已有 `expert.h5` 中：

- 读取所有 `mouse_dx/dy`；
- 统计基本分布；
- 自动生成带冗余（略大于录制范围）的对称挡位列表。

使用示例：

```bash
python mouse_bins.py --data COD_BC/data/expert.h5 --nx 23 --ny 15
```

脚本会打印推荐的：

```text
MOUSE_X_POSSIBLES = [...]
MOUSE_Y_POSSIBLES = [...]
```

可以直接拷贝回 `COD_BC/config.py`。

---

## 常见现象与调试建议

- **训练 eval 准确率不低，但游戏里鼠标几乎不动**：
  - 检查 `data-info` 中的 `max|dx|/max|dy|`，如果非常小，说明专家本身移动很少；
  - 考虑多录一些“鼠标动作剧烈”的片段，并用 `mouse_bins.py` 更新挡位；
  - 未来可以在 `trainer.py` 的鼠标 CE loss 里加入 class weight，对大位移档位升权。

- **推理日志中 `move=(0.0,0.0)` 一直不变**：
  - 表明模型几乎总选中心档位；
  - 优先从数据与损失权重层面优化，而非改推理代码。

- **推理日志中 `move` 经常非 0，但屏幕不动**：
  - 检查 COD 窗口是否在活动显示器上且获得焦点；
  - 检查系统或游戏是否屏蔽 `pyautogui` 的鼠标事件。

本 README 只覆盖主要使用方式和结构，更多细节与设计动机可参考 `COD_BC/docs/训练流程.md`。