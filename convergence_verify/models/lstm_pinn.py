"""
LSTM-PINN 对手建模模块 — 目标无人机轨迹预测网络。

架构（论文表 4.3）：
  输入: 10步历史 [x, y, vx, vy]  →  (batch, 10, 4)
  LSTM Layer 1 (hidden=128) + ReLU + Dropout(0.2)
  LSTM Layer 2 (hidden=64) + ReLU + Dropout(0.2)
  FC (32) + ReLU
  输出: 预测下一时刻位置 [x, y]  →  (batch, 2)

损失函数（论文公式 4.28-4.31）：
  L_total = λ_data * L_data + λ_phy * L_phy + λ_smooth * L_smooth
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field


@dataclass
class LSTMConfig:
    """LSTM-PINN 模型超参数（对齐论文表 4.3）。"""
    seq_length: int = 10              # 历史时间窗口
    input_dim: int = 4                # [x, y, vx, vy]
    lstm_hidden: list = field(default_factory=lambda: [128, 64])
    fc_hidden: int = 32
    output_dim: int = 2               # 预测 [x_hat, y_hat]
    dropout: float = 0.2

    # 损失权重（论文公式 4.31）
    lam_data: float = 0.7
    lam_phy: float = 0.25
    lam_smooth: float = 0.05

    # 训练
    lr: float = 0.001
    batch_size: int = 64
    max_epochs: int = 100
    early_stop_patience: int = 15


class LSTMPINN(nn.Module):
    """
    堆叠式 LSTM 轨迹预测网络，端到端输出下一时刻位置。

    输入:  (B, seq_len, 4)  →  [x_t, y_t, vx_t, vy_t] × seq_len
    输出:  (B, 2)           →  [x_{t+1}, y_{t+1}]
    """

    def __init__(self, config: LSTMConfig = None):
        super().__init__()
        if config is None:
            config = LSTMConfig()
        self.cfg = config

        # ---- 堆叠 LSTM ----
        self.lstm1 = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.lstm_hidden[0],
            num_layers=1,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(config.dropout)

        self.lstm2 = nn.LSTM(
            input_size=config.lstm_hidden[0],
            hidden_size=config.lstm_hidden[1],
            num_layers=1,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(config.dropout)

        # ---- 全连接层 ----
        self.fc1 = nn.Linear(config.lstm_hidden[1], config.fc_hidden)
        self.fc2 = nn.Linear(config.fc_hidden, config.output_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor  (B, seq_len, input_dim)
            历史状态序列。

        Returns
        -------
        torch.Tensor  (B, 2)
            预测的下一时刻位置 [x, y]。
        """
        # LSTM Layer 1
        out, _ = self.lstm1(x)
        out = self.relu(out)
        out = self.dropout1(out)

        # LSTM Layer 2
        out, _ = self.lstm2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        # 取最后一个时间步的隐藏状态
        out = out[:, -1, :]                        # (B, lstm_hidden[1])

        # FC
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)                        # (B, 2)

        return out

    def predict(self, history_seq):
        """
        推理接口：输入 numpy 历史序列，返回 numpy 预测位置。

        Parameters
        ----------
        history_seq : np.ndarray  (seq_len, 4) or (B, seq_len, 4)
            历史状态序列。

        Returns
        -------
        np.ndarray  (2,) or (B, 2)
            预测位置 [x_pred, y_pred]。
        """
        self.eval()
        single = (history_seq.ndim == 2)
        if single:
            history_seq = history_seq[np.newaxis, ...]          # (1, seq_len, 4)

        x = torch.as_tensor(history_seq, dtype=torch.float32)
        with torch.no_grad():
            pred = self.forward(x)

        if single:
            return pred.squeeze(0).numpy()
        return pred.numpy()


class LSTMPINNLoss:
    """
    LSTM-PINN 复合损失函数。

    L_total = λ_data * L_data  +  λ_phy * L_phy  +  λ_smooth * L_smooth

    L_data   — MSE(预测位置, 真实位置)
    L_phy    — 物理一致性残差：基于预测速度外推的位置应与预测位置一致
    L_smooth — 加速度平滑约束：惩罚相邻时刻加速度的突变
    """

    def __init__(self, config: LSTMConfig = None):
        if config is None:
            config = LSTMConfig()
        self.lam_data = config.lam_data
        self.lam_phy = config.lam_phy
        self.lam_smooth = config.lam_smooth
        self.mse = nn.MSELoss()

    def compute(self, pred_pos, true_pos, x_seq):
        """
        计算总损失及其分量。

        Parameters
        ----------
        pred_pos : torch.Tensor  (B, 2)
            网络预测的下一时刻位置。
        true_pos : torch.Tensor  (B, 2)
            真实的下一时刻位置标签。
        x_seq : torch.Tensor  (B, seq_len, 4)
            输入的完整历史序列（用于计算物理残差和平滑损失）。

        Returns
        -------
        total_loss : torch.Tensor  scalar
        losses : dict
            {"L_data": float, "L_phy": float, "L_smooth": float, "L_total": float}
        """
        B = x_seq.size(0)

        # ---- L_data: 预测位置 vs 真实位置 ----
        L_data = self.mse(pred_pos, true_pos)

        # ---- L_phy: 物理一致性残差 ----
        # 利用最后一步的预测速度和位置推导 p_phy = p_{-1} + v_pred * dt
        # 但本网络只输出 [x, y]，没有显式输出速度。
        # 论文方法：在输入序列上计算速度，然后验证位移-速度耦合。
        # 简化：用序列最后两步差分近似瞬时速度
        if x_seq.size(1) >= 2:
            # 最后两个时间步的位置变化近似速度
            v_est = (x_seq[:, -1, :2] - x_seq[:, -2, :2]) / 0.05   # dt=0.05
            p_phy = x_seq[:, -1, :2] + v_est * 0.05                # 基于速度外推
            L_phy = self.mse(pred_pos, p_phy)
        else:
            L_phy = torch.tensor(0.0, device=pred_pos.device)

        # ---- L_smooth: 加速度平滑约束 ----
        # 计算序列各时刻的加速度二阶差分，惩罚突变
        if x_seq.size(1) >= 4:
            # v_t ≈ (p_t - p_{t-1}) / dt
            positions = x_seq[:, :, :2]                             # (B, T, 2)
            v_seq = (positions[:, 1:, :] - positions[:, :-1, :])   # (B, T-1, 2)
            a_seq = v_seq[:, 1:, :] - v_seq[:, :-1, :]             # (B, T-2, 2) 加速度差分
            L_smooth = a_seq.pow(2).mean()
        else:
            L_smooth = torch.tensor(0.0, device=pred_pos.device)

        # ---- 加权总和 ----
        L_total = (self.lam_data * L_data +
                   self.lam_phy * L_phy +
                   self.lam_smooth * L_smooth)

        losses = {
            "L_data": L_data.item(),
            "L_phy": L_phy.item(),
            "L_smooth": L_smooth.item(),
            "L_total": L_total.item(),
        }

        return L_total, losses
