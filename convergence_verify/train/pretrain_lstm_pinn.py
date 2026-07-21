#!/usr/bin/env python3
"""
LSTM-PINN 对手建模模块预训练脚本。

使用 UAVEnv 采集的轨迹数据训练 LSTM-PINN 网络，并评估预测精度。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.lstm_pinn import LSTMPINN, LSTMPINNLoss, LSTMConfig
from models.data_utils import TrajectoryCollector, TrajectoryDataset
from env.uav_env import UAVEnv

# 评估指标：MAE
def compute_mae(model, dataloader):
    model.eval()
    total_abs_error = 0.0
    total_samples = 0
    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            pred = model(X_batch)
            abs_error = (pred - Y_batch).abs().sum(dim=1)
            total_abs_error += abs_error.sum().item()
            total_samples += X_batch.size(0)
    return total_abs_error / total_samples


def train_epoch(model, optimizer, loss_fn, dataloader):
    model.train()
    epoch_losses = {"L_data": 0.0, "L_phy": 0.0, "L_smooth": 0.0, "L_total": 0.0}
    n_batches = 0

    for X_batch, Y_batch in dataloader:
        pred = model(X_batch)
        total_loss, losses = loss_fn.compute(pred, Y_batch, X_batch)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        for k in epoch_losses:
            epoch_losses[k] += losses[k]
        n_batches += 1

    return {k: v / n_batches for k, v in epoch_losses.items()}


def pretrain(config: LSTMConfig, train_loader, val_loader, device="cpu"):
    """主训练循环，返回训练好的模型和训练历史。"""
    model = LSTMPINN(config).to(device)
    loss_fn = LSTMPINNLoss(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    history = {
        "train_loss": [], "val_loss": [],
        "train_mae": [], "val_mae": [],
    }
    best_val_mae = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state = None

    pbar = tqdm(range(1, config.max_epochs + 1), desc="Training LSTM-PINN")
    for epoch in pbar:
        # 训练
        train_losses = train_epoch(model, optimizer, loss_fn, train_loader)
        train_mae = compute_mae(model, train_loader)

        # 验证
        val_losses = train_epoch(model, optimizer, loss_fn, val_loader)  # 仅前向
        # 重新计算验证损失（不用 optimizer step 的版本）
        model.eval()
        val_total = 0.0
        n_val = 0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                pred = model(X_batch)
                total_loss, _ = loss_fn.compute(pred, Y_batch, X_batch)
                val_total += total_loss.item()
                n_val += 1
        val_mae = compute_mae(model, val_loader)

        # 记录
        history["train_loss"].append(train_losses["L_total"])
        history["val_loss"].append(val_total / n_val)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)

        pbar.set_postfix({
            "L_data": f"{train_losses['L_data']:.4f}",
            "L_phy": f"{train_losses['L_phy']:.4f}",
            "ValMAE": f"{val_mae:.4f}",
        })

        # 早停
        if val_mae < best_val_mae - 1e-5:
            best_val_mae = val_mae
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= config.early_stop_patience:
            print(f"\n早停于 epoch {epoch}，最佳 val MAE = {best_val_mae:.4f} (epoch {best_epoch})")
            break

    # 加载最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_val_mae


def main():
    """
    完整流程:
      1. 创建环境
      2. 采集轨迹数据
      3. 构建 Dataset + 划分
      4. 训练 LSTM-PINN
      5. 测试集评估
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=200, help="采集回合数")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-model", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- 1. 创建环境 ----
    print("\n" + "=" * 60)
    print("Step 1: 创建 UAV 仿真环境")
    print("=" * 60)
    env = UAVEnv()

    # ---- 2. 采集轨迹 ----
    print("\n" + "=" * 60)
    print("Step 2: 采集目标无人机轨迹数据")
    print("=" * 60)
    collector = TrajectoryCollector(env)
    data_dir = args.data_dir or os.path.join(
        os.path.dirname(__file__), "..", "data", "trajectories"
    )

    all_trajs = []
    for policy in ["random", "apf"]:
        # 临时修改环境以切换策略
        env.escape_policy_name = policy
        episodes = collector.collect_episodes(
            n_episodes=args.n_episodes // 2,
            max_steps=200,
            policy_name=policy,
            save_dir=data_dir,
        )
        # 提取目标轨迹
        for ep in episodes:
            if ep["target"].shape[0] > 10:
                all_trajs.append(ep["target"])

    print(f"\n共采集 {len(all_trajs)} 条有效轨迹")

    # ---- 3. 构建 Dataset ----
    print("\n" + "=" * 60)
    print("Step 3: 构建 PyTorch Dataset")
    print("=" * 60)
    dataset = TrajectoryDataset(all_trajs, seq_length=10)
    train_ds, val_ds, test_ds = dataset.split(ratios=(0.7, 0.15, 0.15))

    print(f"总样本: {len(dataset)}")
    print(f"训练: {len(train_ds)}, 验证: {len(val_ds)}, 测试: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # ---- 4. 训练 ----
    print("\n" + "=" * 60)
    print("Step 4: 训练 LSTM-PINN")
    print("=" * 60)
    config = LSTMConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
    )
    model, history, best_val_mae = pretrain(config, train_loader, val_loader, device)

    # ---- 5. 测试集评估 ----
    print("\n" + "=" * 60)
    print("Step 5: 测试集评估")
    print("=" * 60)
    test_mae = compute_mae(model, test_loader)
    print(f"测试集 MAE: {test_mae:.4f}")

    # ---- 保存模型 ----
    if args.save_model:
        save_path = args.save_model
    else:
        save_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "lstm_pinn_model.pt"
        )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "best_val_mae": best_val_mae,
        "test_mae": test_mae,
        "history": history,
    }, save_path)
    print(f"模型已保存至 {save_path}")

    # ---- 预测样例 ----
    print("\n" + "=" * 60)
    print("预测样例")
    print("=" * 60)
    sample_X, sample_Y = test_ds[0]
    pred = model(sample_X.unsqueeze(0))
    print(f"历史末帧位置: {sample_X[-1, :2].numpy()}")
    print(f"预测下一位置:  {pred.squeeze(0).detach().numpy()}")
    print(f"真实下一位置:  {sample_Y.numpy()}")
    print(f"绝对误差:      {np.abs(pred.squeeze(0).detach().numpy() - sample_Y.numpy())}")

    return model, history


if __name__ == "__main__":
    main()
