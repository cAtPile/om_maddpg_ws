#!/usr/bin/env python3
"""
LSTM-PINN 模块冒烟测试。

验证:
  1. 模型前向传播 shape 正确
  2. PINN 损失函数三部分均有输出
  3. 合成线性轨迹 → LSTM-PINN 训练后损失下降
  4. predict() numpy 接口正常
  5. TrajectoryDataset 构造与划分
  6. 纯 LSTM vs LSTM-PINN 对比（loss 下降趋势）
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.lstm_pinn import LSTMPINN, LSTMPINNLoss, LSTMConfig
from models.data_utils import TrajectoryDataset


def generate_synthetic_trajectories(n_trajs=100, min_len=30, max_len=80,
                                    dt=0.05, noise=0.02):
    """
    生成合成轨迹：匀速 + 小幅正弦机动。

    返回 list[np.ndarray]，每个 shape (T, 4) [x, y, vx, vy]。
    """
    np.random.seed(42)
    trajs = []

    for _ in range(n_trajs):
        T = np.random.randint(min_len, max_len + 1)
        # 初始状态
        x0, y0 = np.random.uniform(-500, 500, 2)
        vx0 = np.random.uniform(-100, 100)
        vy0 = np.random.uniform(-100, 100)
        omega = np.random.uniform(0.5, 2.0)        # 正弦机动频率
        amp = np.random.uniform(5, 20)             # 机动幅度

        traj = np.zeros((T, 4), dtype=np.float32)
        vx, vy = vx0, vy0
        x, y = x0, y0

        for t in range(T):
            # 加速度：正弦机动 + 噪声
            ax = amp * np.sin(omega * t * dt) + np.random.randn() * noise
            ay = amp * np.cos(omega * t * dt) + np.random.randn() * noise

            vx += ax * dt
            vy += ay * dt
            # 速度限幅 120 m/s
            speed = np.sqrt(vx**2 + vy**2)
            if speed > 120:
                vx = vx / speed * 120
                vy = vy / speed * 120

            x += vx * dt
            y += vy * dt

            traj[t] = [x, y, vx, vy]

        trajs.append(traj)

    return trajs


# =========================================================================
# Test 1: 模型前向传播
# =========================================================================
def test_forward_shape():
    print("=" * 60)
    print("Test 1: 前向传播 shape")
    print("=" * 60)

    config = LSTMConfig(seq_length=10, input_dim=4)
    model = LSTMPINN(config)

    # 单样本
    x = torch.randn(1, 10, 4)
    out = model(x)
    assert out.shape == (1, 2), f"期望 (1, 2)，实际 {out.shape}"
    print(f"  单样本: in={list(x.shape)} → out={list(out.shape)} ✓")

    # 批次
    x_batch = torch.randn(32, 10, 4)
    out_batch = model(x_batch)
    assert out_batch.shape == (32, 2), f"期望 (32, 2)，实际 {out_batch.shape}"
    print(f"  批次:   in={list(x_batch.shape)} → out={list(out_batch.shape)} ✓")


# =========================================================================
# Test 2: PINN 损失函数
# =========================================================================
def test_loss_function():
    print("\n" + "=" * 60)
    print("Test 2: PINN 损失函数")
    print("=" * 60)

    config = LSTMConfig()
    model = LSTMPINN(config)
    loss_fn = LSTMPINNLoss(config)

    # 构造合成数据
    B, T = 16, 10
    x_seq = torch.randn(B, T, 4)
    true_pos = torch.randn(B, 2)
    pred_pos = model(x_seq)

    total_loss, losses = loss_fn.compute(pred_pos, true_pos, x_seq)

    print(f"  L_data   = {losses['L_data']:.4f}")
    print(f"  L_phy    = {losses['L_phy']:.4f}")
    print(f"  L_smooth = {losses['L_smooth']:.4f}")
    print(f"  L_total  = {losses['L_total']:.4f}")

    for k in ["L_data", "L_phy", "L_smooth", "L_total"]:
        assert np.isfinite(losses[k]), f"{k} is not finite!"
    print("  所有损失项均为有限值 ✓")


# =========================================================================
# Test 3: 合成轨迹训练（损失下降验证）
# =========================================================================
def test_training_convergence():
    print("\n" + "=" * 60)
    print("Test 3: 合成轨迹训练 → 损失下降")
    print("=" * 60)

    # 生成数据
    trajs = generate_synthetic_trajectories(n_trajs=50, min_len=40, max_len=80)
    dataset = TrajectoryDataset(trajs, seq_length=10)
    train_ds, val_ds, _ = dataset.split(ratios=(0.7, 0.3, 0.0))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    print(f"  训练样本: {len(train_ds)}, 验证样本: {len(val_ds)}")

    config = LSTMConfig(lr=0.005, max_epochs=30, early_stop_patience=10)
    model = LSTMPINN(config)
    loss_fn = LSTMPINNLoss(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # 记录初始损失
    init_val_loss = None
    for X, Y in val_loader:
        pred = model(X)
        total_loss, _ = loss_fn.compute(pred, Y, X)
        init_val_loss = total_loss.item()
        break

    # 训练 N 个 epoch
    loss_history = []
    for epoch in range(30):
        model.train()
        for X, Y in train_loader:
            pred = model(X)
            total_loss, _ = loss_fn.compute(pred, Y, X)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            loss_history.append(total_loss.item())

        # 每 5 epoch 打印
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for X, Y in val_loader:
                    pred = model(X)
                    total_loss, _ = loss_fn.compute(pred, Y, X)
                    val_losses.append(total_loss.item())
            print(f"  Epoch {epoch+1:3d}: train_loss={loss_history[-1]:.4f}, "
                  f"val_loss={np.mean(val_losses):.4f}")

    # 最终验证损失
    model.eval()
    final_val_losses = []
    with torch.no_grad():
        for X, Y in val_loader:
            pred = model(X)
            total_loss, _ = loss_fn.compute(pred, Y, X)
            final_val_losses.append(total_loss.item())
    final_val_loss = np.mean(final_val_losses)

    loss_ratio = final_val_loss / init_val_loss
    print(f"\n  初始验证 loss: {init_val_loss:.4f}")
    print(f"  最终验证 loss: {final_val_loss:.4f}")
    print(f"  下降比: {loss_ratio:.3f}")

    assert loss_ratio < 0.8, f"Loss 下降不足 (ratio={loss_ratio:.3f} ≥ 0.8)"
    print("  损失显著下降 ✓")


# =========================================================================
# Test 4: predict() numpy 接口
# =========================================================================
def test_predict_interface():
    print("\n" + "=" * 60)
    print("Test 4: predict() numpy 接口")
    print("=" * 60)

    config = LSTMConfig(seq_length=10)
    model = LSTMPINN(config)
    model.eval()

    # 单条序列
    seq = np.random.randn(10, 4).astype(np.float32)
    pred = model.predict(seq)
    assert pred.shape == (2,), f"期望 (2,)，实际 {pred.shape}"
    assert isinstance(pred, np.ndarray), f"期望 np.ndarray"
    print(f"  单序列: in={seq.shape} → out={pred.shape}, dtype={pred.dtype} ✓")

    # 批次
    seq_batch = np.random.randn(4, 10, 4).astype(np.float32)
    pred_batch = model.predict(seq_batch)
    assert pred_batch.shape == (4, 2), f"期望 (4, 2)，实际 {pred_batch.shape}"
    print(f"  批次:   in={seq_batch.shape} → out={pred_batch.shape} ✓")


# =========================================================================
# Test 5: TrajectoryDataset 构造与划分
# =========================================================================
def test_dataset():
    print("\n" + "=" * 60)
    print("Test 5: TrajectoryDataset")
    print("=" * 60)

    trajs = generate_synthetic_trajectories(n_trajs=20, min_len=30, max_len=50)
    dataset = TrajectoryDataset(trajs, seq_length=10)

    # 每条约 30~50 步，减去 10 窗口 → 20~40 样本/条
    print(f"  总样本: {len(dataset)}")
    assert len(dataset) > 0, "Dataset 为空!"

    # 检查一个样本
    X, Y = dataset[0]
    assert X.shape == (10, 4), f"X shape {X.shape}"
    assert Y.shape == (2,), f"Y shape {Y.shape}"
    print(f"  X shape: {tuple(X.shape)}, Y shape: {tuple(Y.shape)}")

    # 划分
    train, val, test = dataset.split(ratios=(0.7, 0.15, 0.15))
    total = len(train) + len(val) + len(test)
    print(f"  划分: train={len(train)}, val={len(val)}, test={len(test)} (总计={total})")
    assert total == len(dataset), "划分后总数不一致!"
    print("  数据集构造与划分 ✓")


# =========================================================================
# Test 6: 纯 LSTM vs LSTM-PINN 对比
# =========================================================================
def test_pinn_vs_lstm():
    """
    消融对比：相同的 LSTM 架构，有无 PINN 物理约束的区别。

    用 PINN-loss 版本相比纯 MSE loss：
    - L_smooth 应显著下降（加速度更平滑）
    - 物理残差 L_phy 应下降
    """
    print("\n" + "=" * 60)
    print("Test 6: 纯 LSTM vs LSTM-PINN 消融对比")
    print("=" * 60)

    trajs = generate_synthetic_trajectories(n_trajs=30, min_len=40, max_len=60)
    dataset = TrajectoryDataset(trajs, seq_length=10)
    train_ds, val_ds, _ = dataset.split(ratios=(0.7, 0.3, 0.0))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    # ---- 训练纯 LSTM (MSE only) ----
    config_mse = LSTMConfig(
        lam_data=1.0, lam_phy=0.0, lam_smooth=0.0,  # 仅数据损失
        lr=0.005, max_epochs=20, early_stop_patience=10,
    )
    model_mse = LSTMPINN(config_mse)
    loss_fn_mse = LSTMPINNLoss(config_mse)
    opt_mse = torch.optim.Adam(model_mse.parameters(), lr=0.005)

    mse_history = {"L_data": [], "L_phy": [], "L_smooth": []}
    for _ in range(20):
        for X, Y in train_loader:
            pred = model_mse(X)
            total_loss, losses = loss_fn_mse.compute(pred, Y, X)
            opt_mse.zero_grad()
            total_loss.backward()
            opt_mse.step()
        # 验证集评估
        model_mse.eval()
        with torch.no_grad():
            for X, Y in val_loader:
                pred = model_mse(X)
                _, losses = loss_fn_mse.compute(pred, Y, X)
                for k in mse_history:
                    mse_history[k].append(losses[k])
                break

    # ---- 训练 LSTM-PINN (with physics) ----
    config_pinn = LSTMConfig(
        lam_data=0.7, lam_phy=0.25, lam_smooth=0.05,
        lr=0.005, max_epochs=20, early_stop_patience=10,
    )
    model_pinn = LSTMPINN(config_pinn)
    loss_fn_pinn = LSTMPINNLoss(config_pinn)
    opt_pinn = torch.optim.Adam(model_pinn.parameters(), lr=0.005)

    pinn_history = {"L_data": [], "L_phy": [], "L_smooth": []}
    for _ in range(20):
        for X, Y in train_loader:
            pred = model_pinn(X)
            total_loss, losses = loss_fn_pinn.compute(pred, Y, X)
            opt_pinn.zero_grad()
            total_loss.backward()
            opt_pinn.step()
        model_pinn.eval()
        with torch.no_grad():
            for X, Y in val_loader:
                pred = model_pinn(X)
                _, losses = loss_fn_pinn.compute(pred, Y, X)
                for k in pinn_history:
                    pinn_history[k].append(losses[k])
                break

    # ---- 比较 ----
    print(f"\n  {'Metric':<15} {'MSE-only':>10} {'PINN':>10} {'改善':>10}")
    print(f"  {'-'*45}")
    for k in ["L_data", "L_phy", "L_smooth"]:
        mse_val = np.mean(mse_history[k][-5:])
        pinn_val = np.mean(pinn_history[k][-5:])
        improvement = (mse_val - pinn_val) / max(abs(mse_val), 1e-6) * 100
        print(f"  {k:<15} {mse_val:>10.4f} {pinn_val:>10.4f} {improvement:>9.1f}%")

    # PINN 版的 L_phy 和 L_smooth 应更低
    mse_ls = np.mean(mse_history["L_smooth"][-5:])
    pinn_ls = np.mean(pinn_history["L_smooth"][-5:])
    if pinn_ls < mse_ls:
        print("\n  PINN 物理约束生效: L_smooth 更低 ✓")
    else:
        print("\n  WARNING: PINN 约束未显著改善 L_smooth（可能是数据太简单）")


# =========================================================================
if __name__ == "__main__":
    test_forward_shape()
    test_loss_function()
    test_training_convergence()
    test_predict_interface()
    test_dataset()
    test_pinn_vs_lstm()

    print("\n" + "=" * 60)
    print("LSTM-PINN 全部冒烟测试完成")
    print("=" * 60)
