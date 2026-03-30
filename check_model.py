"""
检测 best_model.pt 的鼠标移动输出分布。
从 checkpoint 加载模型，用随机输入做前向推理，统计鼠标 X/Y 各档位的 softmax 概率分布。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'COD_BC'))

import torch
import numpy as np

from COD_BC.src.model import build_model
from COD_BC.config import (
    MOUSE_X_POSSIBLES, MOUSE_Y_POSSIBLES,
    N_MOUSE_X, N_MOUSE_Y,
    IMG_HEIGHT, IMG_WIDTH,
)

CHECKPOINT = r'd:\Code\BC_COD\models\best_model.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_SAMPLES = 20   # 随机样本数
SEQ_LEN = 4     # 检测用短序列

print(f'Device: {DEVICE}')
print(f'Loading checkpoint: {CHECKPOINT}')

model = build_model(DEVICE)
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
if 'model_state' in ckpt:
    model.load_state_dict(ckpt['model_state'])
    meta = {k: v for k, v in ckpt.items() if k != 'model_state'}
    print(f'Checkpoint meta: {meta}')
else:
    model.load_state_dict(ckpt)
model.eval()
print('Model loaded.\n')

# 随机输入：(1, SEQ_LEN, 3, H, W)
all_pred_x, all_pred_y = [], []
all_dx, all_dy = [], []

with torch.no_grad():
    for i in range(N_SAMPLES):
        obs = torch.rand(1, SEQ_LEN, 3, IMG_HEIGHT, IMG_WIDTH, device=DEVICE)
        mx_logits, my_logits, btn_logits, key_logits, _ = model(obs)
        # 取最后一个时间步
        mx_prob = torch.softmax(mx_logits[0, -1], dim=-1).cpu().numpy()
        my_prob = torch.softmax(my_logits[0, -1], dim=-1).cpu().numpy()
        pred_x_idx = int(np.argmax(mx_prob))
        pred_y_idx = int(np.argmax(my_prob))
        all_pred_x.append(pred_x_idx)
        all_pred_y.append(pred_y_idx)
        all_dx.append(MOUSE_X_POSSIBLES[pred_x_idx])
        all_dy.append(MOUSE_Y_POSSIBLES[pred_y_idx])

print('=== 鼠标 X 输出统计 ===')
print(f'  预测档位索引分布: {sorted(set(all_pred_x))}')
print(f'  对应像素位移: {[MOUSE_X_POSSIBLES[i] for i in sorted(set(all_pred_x))]}')
print(f'  均值: {np.mean(all_dx):.3f}  标准差: {np.std(all_dx):.3f}')
zero_x = sum(1 for d in all_dx if d == 0.0)
print(f'  输出 dx=0 的次数: {zero_x}/{N_SAMPLES} ({100*zero_x/N_SAMPLES:.0f}%)')

print()
print('=== 鼠标 Y 输出统计 ===')
print(f'  预测档位索引分布: {sorted(set(all_pred_y))}')
print(f'  对应像素位移: {[MOUSE_Y_POSSIBLES[i] for i in sorted(set(all_pred_y))]}')
print(f'  均值: {np.mean(all_dy):.3f}  标准差: {np.std(all_dy):.3f}')
zero_y = sum(1 for d in all_dy if d == 0.0)
print(f'  输出 dy=0 的次数: {zero_y}/{N_SAMPLES} ({100*zero_y/N_SAMPLES:.0f}%)')

print()
if zero_x == N_SAMPLES and zero_y == N_SAMPLES:
    print('[结论] 模型鼠标输出全为 0 —— 可能存在模式崩塌（collapse），建议检查训练数据或损失权重。')
elif zero_x > N_SAMPLES * 0.8 or zero_y > N_SAMPLES * 0.8:
    print('[结论] 模型鼠标输出大多为 0 —— 轻微崩塌，建议增大鼠标损失权重或检查档位分布。')
else:
    print('[结论] 模型有非零鼠标移动输出，档位分布正常。')

# 打印最后一次推理的完整 softmax 分布（前5高概率档位）
print()
print('=== 最后一次推理：X 头 top-5 档位 ===')
with torch.no_grad():
    obs = torch.rand(1, SEQ_LEN, 3, IMG_HEIGHT, IMG_WIDTH, device=DEVICE)
    mx_logits, my_logits, _, _, _ = model(obs)
    mx_prob = torch.softmax(mx_logits[0, -1], dim=-1).cpu().numpy()
    my_prob = torch.softmax(my_logits[0, -1], dim=-1).cpu().numpy()

top5_x = np.argsort(mx_prob)[::-1][:5]
for idx in top5_x:
    print(f'  idx={idx:2d}  dx={MOUSE_X_POSSIBLES[idx]:8.2f}  prob={mx_prob[idx]:.4f}')

print()
print('=== 最后一次推理：Y 头 top-5 档位 ===')
top5_y = np.argsort(my_prob)[::-1][:5]
for idx in top5_y:
    print(f'  idx={idx:2d}  dy={MOUSE_Y_POSSIBLES[idx]:8.2f}  prob={my_prob[idx]:.4f}')
