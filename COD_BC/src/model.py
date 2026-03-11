# 该文件基于 https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning 的模型与流程思路改写
# 原作者：Tim Pearce, Jun Zhu
# 衍生项目：COD_BC (https://github.com/ImJX-Xu/COD_Behavioural_Cloning)，采用 PyTorch 重写
from typing import Optional, Tuple

import torch
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from config import N_MOUSE_X, N_MOUSE_Y, IMG_HEIGHT, IMG_WIDTH, USE_SEQUENCE_LSTM


class ConvLSTMCell2D(nn.Module):
    """单层 ConvLSTM Cell：输入 (B, C_in, H, W)，隐状态 (h, c) 各 (B, C_hidden, H, W)。"""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=padding,
        )

    def forward(
        self,
        x: torch.Tensor,
        cur_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if cur_state is None:
            h = torch.zeros(x.size(0), self.hidden_dim, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            c = torch.zeros_like(h)
        else:
            h, c = cur_state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM2D(nn.Module):
    """
    单层 ConvLSTM：输入 (B, T, C, H, W)，可选 hidden=(h, c) 各 (B, C_hidden, H, W)。
    输出 (B, T, C_hidden, H, W) 与 next_hidden=(h_n, c_n)。
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.cell = ConvLSTMCell2D(input_dim, hidden_dim, kernel_size)
        self.hidden_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        b, t, c, h, w = x.shape
        if hidden is None:
            h_cur = torch.zeros(b, self.hidden_dim, h, w, device=x.device, dtype=x.dtype)
            c_cur = torch.zeros_like(h_cur)
        else:
            h_cur, c_cur = hidden
        out_list = []
        for i in range(t):
            h_cur, c_cur = self.cell(x[:, i], (h_cur, c_cur))
            out_list.append(h_cur)
        out = torch.stack(out_list, dim=1)
        return out, (h_cur, c_cur)


class EfficientNetLSTMBCModel(nn.Module):
    """
    EfficientNetB0 features + ConvLSTM2D + LSTM + 4 头（与 Example 风格一致）。
    输入: obs [B, T, 3, IMG_HEIGHT, IMG_WIDTH]，如 (B, T, 3, 90, 410)。
    支持 hidden=((conv_h, conv_c), (lstm_h, lstm_c)) 做 stateful 推理。
    """

    def __init__(
        self,
        lstm_hidden_size: int = 512,
        lstm_layers: int = 2,
        convlstm_hidden: int = 256,
        pretrained: bool = True,
        use_sequence_lstm: bool = USE_SEQUENCE_LSTM,
    ):
        super().__init__()
        self.convlstm_hidden = convlstm_hidden
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_layers
        self.use_sequence_lstm = use_sequence_lstm
        self.n_mouse_x = N_MOUSE_X
        self.n_mouse_y = N_MOUSE_Y

        # 与 Example 一致：使用 intermediate backbone（截断最后 conv_head），减少显存
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        # features[:-2] 去掉 conv_head+bn1，输出 192 通道，显存约为完整版的 1/4
        self.backbone = nn.Sequential(*list(backbone.features.children())[:-2])
        self.feat_reduce = nn.Conv2d(192, convlstm_hidden, 1)
        self.convlstm = ConvLSTM2D(convlstm_hidden, convlstm_hidden, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        if self.use_sequence_lstm:
            self.lstm = nn.LSTM(
                input_size=convlstm_hidden,
                hidden_size=lstm_hidden_size,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=0.2,
            )
            fc_in_dim = lstm_hidden_size
        else:
            self.lstm = None
            fc_in_dim = convlstm_hidden
        self.fc1 = nn.Linear(fc_in_dim, 256)
        self.relu = nn.ReLU(inplace=True)
        self.fc_mouse_x = nn.Linear(256, N_MOUSE_X)
        self.fc_mouse_y = nn.Linear(256, N_MOUSE_Y)
        self.fc_buttons = nn.Linear(256, 2)
        self.fc_keys = nn.Linear(256, 6)

    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """
        hidden: ((conv_h, conv_c), (lstm_h, lstm_c)) 或 None。
        返回 mouse_x_logits, mouse_y_logits, mouse_buttons_logits, key_logits, next_hidden。
        """
        b, t, c, h, w = obs.shape
        x = obs.view(b * t, c, h, w)
        x = self.backbone(x)  # (B*T, 192, h', w')
        x = self.feat_reduce(x)  # (B*T, convlstm_hidden, h', w')
        _, _, hf, wf = x.shape
        x = x.view(b, t, self.convlstm_hidden, hf, wf)

        conv_hidden = hidden[0] if hidden is not None else None
        x, conv_next = self.convlstm(x, conv_hidden)
        b2, t2, c2, hf2, wf2 = x.shape
        x = x.reshape(b2 * t2, c2, hf2, wf2)
        x = self.pool(x)  # (B*T, C, 1, 1)
        x = x.view(b2, t2, c2)

        lstm_next = None
        if self.use_sequence_lstm:
            lstm_hidden = hidden[1] if hidden is not None else None
            lstm_out, lstm_next = self.lstm(x, lstm_hidden)
        else:
            lstm_out = x
        h = self.relu(self.fc1(lstm_out))

        mouse_x_logits = self.fc_mouse_x(h)
        mouse_y_logits = self.fc_mouse_y(h)
        mouse_buttons_logits = self.fc_buttons(h)
        key_logits = self.fc_keys(h)
        next_hidden = (conv_next, lstm_next)
        return mouse_x_logits, mouse_y_logits, mouse_buttons_logits, key_logits, next_hidden


def build_model(device: torch.device):
    model = EfficientNetLSTMBCModel()
    return model.to(device)

