"""
轨迹数据采集与预处理。

TrajectoryCollector : 使用 UAVEnv 生成多策略轨迹数据
TrajectoryDataset   : 将轨迹转为 LSTM-PINN 所需的 (历史序列, 标签) 格式
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryCollector:
    """
    使用 UAVEnv 批量采集目标无人机的轨迹数据。

    支持同时采集随机 / APF 等不同逃逸策略下的轨迹，保存在磁盘。
    """

    def __init__(self, env):
        """
        Parameters
        ----------
        env : UAVEnv
            已初始化的环境实例。
        """
        self.env = env

    def collect_episodes(self, n_episodes=100, max_steps=200,
                         policy_name="random", save_dir=None,
                         record_all=False):
        """
        采集指定数量的完整 episode 轨迹。

        Parameters
        ----------
        n_episodes : int
            采集回合数。
        max_steps : int
            每回合最大步数。
        policy_name : str
            目标逃逸策略 ("random" / "apf")。
        save_dir : str or None
            若提供，保存为 .npz。
        record_all : bool
            是否同时记录追捕者和障碍物状态。

        Returns
        -------
        list[dict]
            每个 dict 包含该回合目标的完整轨迹。
        """
        old_policy = self.env.escape_policy_name
        self._set_target_policy(policy_name)

        episodes = []

        for ep in range(n_episodes):
            obs, info = self.env.reset()

            # 存储本回合目标轨迹
            target_traj = []
            pursuer_trajs = [[] for _ in range(self.env.n_pursuers)] if record_all else None
            obstacle_trajs = None

            for step in range(max_steps):
                # 记录当前状态
                state = self.env.target.get_state()
                target_traj.append([
                    state["x"], state["y"],
                    state["vx"], state["vy"],
                ])

                if record_all:
                    for i, p in enumerate(self.env.pursuers):
                        ps = p.get_state()
                        pursuer_trajs[i].append([ps["x"], ps["y"], ps["vx"], ps["vy"]])

                # 随机动作驱动追捕者（仅做数据采集）
                actions = np.random.uniform(-1, 1, (self.env.n_pursuers, 2))
                obs, rewards, terminated, truncated, info = self.env.step(actions)

                if terminated or truncated:
                    break

            if len(target_traj) >= self.env.max_step:
                episodes.append({
                    "target": np.array(target_traj, dtype=np.float32),
                    "n_steps": len(target_traj),
                    "policy": policy_name,
                })

            if (ep + 1) % max(1, n_episodes // 10) == 0:
                print(f"  [{policy_name}] 已采集 {ep + 1}/{n_episodes} 回合")

        # 恢复原策略
        self._set_target_policy(old_policy)

        # 保存
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"trajectories_{policy_name}.npz")
            arrays = {f"ep_{i}": ep["target"] for i, ep in enumerate(episodes)}
            np.savez_compressed(save_path, **arrays)
            print(f"已保存 {len(episodes)} 条 {policy_name} 轨迹至 {save_path}")

        return episodes

    def _set_target_policy(self, name):
        """临时切换目标策略。"""
        from .target_policy import RandomEscapePolicy, APFEscapePolicy
        if name == "random":
            self.env.target_policy = RandomEscapePolicy(
                v_max=self.env.t_v_max, a_max=self.env.t_a_max
            )
        elif name == "apf":
            self.env.target_policy = APFEscapePolicy(
                v_max=self.env.t_v_max, a_max=self.env.t_a_max,
                bounds=self.env.bounds,
            )
        self.env.escape_policy_name = name


class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset：将原始轨迹数组切分为 (历史序列, 标签) 样本。

    每条轨迹长 T 步，滑动窗口产出 (T - seq_length) 个样本：
      X = [s_{t-seq+1}, ..., s_t]    (seq_len, 4)
      Y = [x_{t+1}, y_{t+1}]         (2,)
    """

    def __init__(self, trajectories, seq_length=10):
        """
        Parameters
        ----------
        trajectories : list[np.ndarray]
            轨迹列表，每个 shape (T_i, 4) 含 [x, y, vx, vy]。
        seq_length : int
            历史窗口长度。
        """
        self.seq_length = seq_length
        self.X = []
        self.Y = []

        for traj in trajectories:
            if traj.shape[0] <= seq_length:
                continue
            T = traj.shape[0]
            for t in range(T - seq_length):
                self.X.append(traj[t : t + seq_length])   # (seq_len, 4)
                self.Y.append(traj[t + seq_length, :2])   # 仅位置 (2,)

        self.X = np.array(self.X, dtype=np.float32)
        self.Y = np.array(self.Y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.as_tensor(self.X[idx]),
            torch.as_tensor(self.Y[idx]),
        )

    @classmethod
    def from_npz_files(cls, file_paths, seq_length=10):
        """
        从 .npz 文件加载轨迹并构建数据集。

        Parameters
        ----------
        file_paths : list[str]
            .npz 文件路径列表。
        seq_length : int

        Returns
        -------
        TrajectoryDataset
        """
        all_trajs = []
        for fp in file_paths:
            data = np.load(fp, allow_pickle=True)
            for key in data.files:
                traj = data[key]
                if traj.shape[0] > seq_length:
                    all_trajs.append(traj)
        return cls(all_trajs, seq_length)

    def split(self, ratios=(0.7, 0.15, 0.15), seed=42):
        """
        按比例随机划分为训练/验证/测试集。

        Returns
        -------
        tuple[TrajectoryDataset, TrajectoryDataset, TrajectoryDataset]
        """
        assert abs(sum(ratios) - 1.0) < 1e-6
        n = len(self)
        indices = np.arange(n)
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)

        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val:]

        def subset(idx_arr):
            ds = TrajectoryDataset.__new__(TrajectoryDataset)
            ds.seq_length = self.seq_length
            ds.X = self.X[idx_arr]
            ds.Y = self.Y[idx_arr]
            return ds

        return subset(train_idx), subset(val_idx), subset(test_idx)
