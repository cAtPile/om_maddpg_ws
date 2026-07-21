"""
无人机集群协同围捕 2D 仿真环境 — 兼容 Gymnasium 接口。

场景: N 架追捕者 vs 1 架目标，含静态/动态圆形障碍物。

状态空间 (每架追捕者):
  [x_i, y_i, vx_i, vy_i,                          # 自身状态 (4)
   x_t, y_t, vx_t, vy_t,                           # 目标状态 (4)
   d_t, angle_t,                                    # 目标相对极坐标 (2)
   lidar_0..lidar_15,                               # LiDAR (16)
   x_j1,y_j1, x_j2,y_j2]                           # 队友位置 (4)
  总计: 30 维

动作空间 (每架追捕者):
  [ax, ay] 连续二维加速度, ∈ [-a_max, a_max]

奖励: R = mu_near*R_near + mu_stage*R_stage + mu_finish*R_finish
        + mu_safe*R_safe + mu_time*R_time
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import yaml
import os

from .kinematics import UAV
from .obstacle import Obstacle
from .lidar import Lidar
from .apollonius import ApolloniusCapture
from .target_policy import RandomEscapePolicy, APFEscapePolicy


class UAVEnv(gym.Env):
    """
    多无人机协同围捕 Gymnasium 环境。

    遵循标准的 reset() -> (obs, info) 和 step() -> (obs, reward, term, trunc, info) 接口。
    """

    metadata = {"render_modes": ["human", "none"], "render_fps": 20}

    def __init__(self, config_path=None, render_mode="none"):
        super().__init__()

        # ---- 加载配置 ----
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "config", "config.yaml"
            )
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.render_mode = render_mode
        self._load_config()

        # ---- 初始化组件 ----
        self._init_uavs()
        self._init_obstacles()
        self._init_lidars()
        self._init_capture_checker()
        self._init_target_policy()

        # ---- Gym 空间定义 ----
        self._setup_spaces()

        # ---- 内部状态 ----
        self.current_step = 0
        self.total_steps_across_episodes = 0
        self.collision_occurred = False

    # ------------------------------------------------------------------
    # 配置加载
    # ------------------------------------------------------------------

    def _load_config(self):
        """从配置 dict 提取所有参数。"""
        c = self.cfg

        # 环境
        env = c["env"]
        self.dt = env["dt"]
        self.max_step = env["max_step"]
        self.bounds = tuple(env["bounds"])
        self.x_min, self.x_max, self.y_min, self.y_max = self.bounds

        # 追捕者
        p = c["pursuer"]
        self.n_pursuers = p["num"]
        self.p_v_max = p["v_max"]
        self.p_a_max = p["a_max"]
        self.p_init_distance = p["init_distance"]

        # 目标
        t = c["target"]
        self.t_v_max = t["v_max"]
        self.t_a_max = t["a_max"]
        self.t_init_pos = tuple(t["init_position"])
        self.t_init_heading = t.get("init_heading", None)
        self.escape_policy_name = t.get("escape_policy", "random")

        # 障碍物
        o = c["obstacle"]
        self.n_obstacles = o["num"]
        self.obs_radius_range = tuple(o["radius_range"])
        self.obs_speed = o["speed"]
        self.obs_dynamic = o["dynamic"]

        # 任务
        tk = c["task"]
        self.d_limit = tk["d_limit"]
        self.d_capture = tk["d_capture"]
        self.r_safe = tk["r_safe"]
        self.r_uav = tk["r_uav"]
        self.angle_coverage_threshold = tk["angle_coverage_threshold"]

        # 奖励权重
        rw = c["reward"]
        self.mu_near = rw["mu_near"]
        self.mu_stage = rw["mu_stage"]
        self.mu_finish = rw["mu_finish"]
        self.mu_safe = rw["mu_safe"]
        self.mu_time = rw["mu_time"]

    # ------------------------------------------------------------------
    # 组件初始化
    # ------------------------------------------------------------------

    def _init_uavs(self):
        """创建追捕者 + 目标 UAV 实例。"""
        self.pursuers = []
        for i in range(self.n_pursuers):
            uav = UAV(
                uav_id=f"pursuer_{i}",
                v_max=self.p_v_max,
                a_max=self.p_a_max,
                bounds=self.bounds,
            )
            self.pursuers.append(uav)

        self.target = UAV(
            uav_id="target",
            v_max=self.t_v_max,
            a_max=self.t_a_max,
            bounds=self.bounds,
        )

    def _init_obstacles(self):
        """创建障碍物列表。"""
        self.obstacles = []
        for i in range(self.n_obstacles):
            radius = np.random.uniform(*self.obs_radius_range)
            obs = Obstacle(
                obstacle_id=i,
                radius=radius,
                speed=self.obs_speed,
                is_dynamic=self.obs_dynamic,
                bounds=self.bounds,
            )
            self.obstacles.append(obs)

    def _init_lidars(self):
        """为每架追捕者创建 LiDAR。"""
        c = self.cfg["lidar"]
        self.lidars = [
            Lidar(n_channels=c["n_channels"], d_max=c["d_max"])
            for _ in range(self.n_pursuers)
        ]

    def _init_capture_checker(self):
        """初始化围捕判定器。"""
        self.capture_checker = ApolloniusCapture(
            d_capture=self.d_capture,
            angle_coverage_threshold=self.angle_coverage_threshold,
        )

    def _init_target_policy(self):
        """根据配置选择目标逃逸策略。"""
        if self.escape_policy_name == "random":
            self.target_policy = RandomEscapePolicy(
                v_max=self.t_v_max, a_max=self.t_a_max
            )
        elif self.escape_policy_name == "apf":
            self.target_policy = APFEscapePolicy(
                v_max=self.t_v_max,
                a_max=self.t_a_max,
                bounds=self.bounds,
            )
        else:
            self.target_policy = RandomEscapePolicy(
                v_max=self.t_v_max, a_max=self.t_a_max
            )

    # ------------------------------------------------------------------
    # 观测/动作空间
    # ------------------------------------------------------------------

    def _setup_spaces(self):
        """定义 Gym 观测空间和动作空间。"""
        # 每架追捕者的观测维度
        n_lidar = self.cfg["lidar"]["n_channels"]  # 16
        self.obs_dim_per_agent = (
            4 +                  # 自身: x, y, vx, vy
            4 +                  # 目标: x, y, vx, vy
            2 +                  # 目标相对极坐标: d, angle
            n_lidar +            # LiDAR: 16 维
            2 * (self.n_pursuers - 1)  # 队友位置: (x, y) each
        )

        # 每个追捕者: Box(low, high) 观测
        self.observation_space = spaces.Dict({
            f"agent_{i}": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.obs_dim_per_agent,),
                dtype=np.float32,
            )
            for i in range(self.n_pursuers)
        })

        # 全局观测（集中式 Critic 用）: 所有追捕者观测的拼接
        self.global_obs_dim = self.obs_dim_per_agent * self.n_pursuers

        # 每个追捕者: 2 维连续动作 [ax, ay]
        self.action_space = spaces.Dict({
            f"agent_{i}": spaces.Box(
                low=-1.0, high=1.0,
                shape=(2,),
                dtype=np.float32,
            )
            for i in range(self.n_pursuers)
        })

        # 单智能体动作空间（便捷访问）
        self.single_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Gym 接口: reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """
        重置环境。

        Returns
        -------
        observations : dict
            每架追捕者的观测。
        info : dict
            额外信息。
        """
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.collision_occurred = False

        # ---- 重置目标 ----
        self.target.reset(
            x=self.t_init_pos[0],
            y=self.t_init_pos[1],
            heading=self.t_init_heading,
        )

        # ---- 重置追捕者（环形分布） ----
        for i, pursuer in enumerate(self.pursuers):
            angle = 2 * np.pi * i / self.n_pursuers + np.random.uniform(-0.2, 0.2)
            px = self.t_init_pos[0] + self.p_init_distance * np.cos(angle)
            py = self.t_init_pos[1] + self.p_init_distance * np.sin(angle)
            # 初始速度较小，朝向目标方向
            toward_target = np.arctan2(-py, -px)
            pursuer.reset(x=px, y=py, heading=toward_target)

        # ---- 重置障碍物 ----
        self._reset_obstacles()

        # ---- 初始 LiDAR 扫描 ----
        self._update_lidars()

        # ---- 构建观测 ----
        observations = self._build_observations()

        info = {
            "target_pos": self.target.get_pos().copy(),
            "pursuer_positions": [p.get_pos().copy() for p in self.pursuers],
            "obstacle_states": [o.get_state() for o in self.obstacles],
        }

        if self.render_mode == "human":
            self.render()

        return observations, info

    def _reset_obstacles(self):
        """随机放置障碍物，避免与无人机位置重叠。"""
        # 收集已被占用的位置
        occupied_positions = [self.target.get_pos()]
        occupied_positions += [p.get_pos() for p in self.pursuers]
        min_separation = 200.0  # 障碍物与无人机最小距离

        for obs in self.obstacles:
            radius = np.random.uniform(*self.obs_radius_range)
            obs.radius = radius

            placed = False
            for _ in range(100):  # 最多尝试 100 次
                x = np.random.uniform(self.x_min + radius + 50,
                                      self.x_max - radius - 50)
                y = np.random.uniform(self.y_min + radius + 50,
                                      self.y_max - radius - 50)
                pos = np.array([x, y])

                # 检查是否与已有物体重叠
                too_close = False
                for occ in occupied_positions:
                    if np.linalg.norm(pos - occ) < min_separation:
                        too_close = True
                        break

                if not too_close:
                    obs.reset(x, y)
                    placed = True
                    break

            if not placed:
                # 兜底：放到远离中心的位置
                angle = np.random.uniform(0, 2 * np.pi)
                dist = np.random.uniform(600, 900)
                obs.reset(
                    dist * np.cos(angle),
                    dist * np.sin(angle),
                )

            occupied_positions.append(obs.get_pos())

    # ------------------------------------------------------------------
    # Gym 接口: step
    # ------------------------------------------------------------------

    def step(self, actions):
        """
        执行一步仿真。

        Parameters
        ----------
        actions : dict or np.ndarray
            若为 dict: {"agent_0": [ax, ay], ...}
            若为 ndarray: shape (n_pursuers, 2)，按 agent_0..agent_N 顺序。

        Returns
        -------
        observations : dict
        rewards : dict
        terminated : bool
        truncated : bool
        info : dict
        """
        self.current_step += 1
        self.total_steps_across_episodes += 1

        # ---- 解析动作 ----
        if isinstance(actions, np.ndarray):
            actions_dict = {
                f"agent_{i}": actions[i] for i in range(self.n_pursuers)
            }
        else:
            actions_dict = actions

        # ---- 1. 目标根据策略选动作并移动 ----
        target_lidar = self._sense_target_lidar()
        target_ax, target_ay = self.target_policy.get_action(
            self.target, self.pursuers, self.obstacles, target_lidar
        )
        self.target.step(target_ax, target_ay, self.dt)

        # ---- 2. 追捕者执行动作并移动 ----
        for i, pursuer in enumerate(self.pursuers):
            normalized_action = actions_dict[f"agent_{i}"]
            # 从 [-1, 1] 缩放到 [-a_max, a_max]
            ax = float(normalized_action[0]) * self.p_a_max
            ay = float(normalized_action[1]) * self.p_a_max
            pursuer.step(ax, ay, self.dt)

        # ---- 3. 障碍物步进 ----
        for obs in self.obstacles:
            obs.step(self.dt)

        # ---- 4. 更新 LiDAR ----
        self._update_lidars()

        # ---- 5. 碰撞检测 ----
        collision = self._check_collisions()

        # ---- 6. 围捕判定 ----
        captured, capture_debug = self._check_capture()

        # ---- 7. 计算奖励 ----
        rewards = self._compute_rewards(collision, captured)

        # ---- 8. 终止条件 ----
        terminated = captured or collision
        truncated = self.current_step >= self.max_step

        # ---- 9. 构建返回 ----
        observations = self._build_observations()

        info = {
            "target_pos": self.target.get_pos().copy(),
            "target_state": self.target.get_state(),
            "pursuer_states": [p.get_state() for p in self.pursuers],
            "obstacle_states": [o.get_state() for o in self.obstacles],
            "capture_debug": capture_debug,
            "collision": collision,
            "captured": captured,
            "current_step": self.current_step,
            "lidar_readings": self._last_lidar_readings,
        }

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminated, truncated, info

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _sense_target_lidar(self):
        """为目标无人机创建临时 LiDAR 进行感知。"""
        target_lidar = Lidar(
            n_channels=self.cfg["lidar"]["n_channels"],
            d_max=self.cfg["lidar"]["d_max"],
        )
        return target_lidar.sense(
            self.target.x, self.target.y,
            self.target.heading, self.obstacles,
        )

    def _update_lidars(self):
        """更新所有追捕者的 LiDAR 读数。"""
        self._last_lidar_readings = []
        for i, (pursuer, lidar) in enumerate(zip(self.pursuers, self.lidars)):
            reading = lidar.sense(
                pursuer.x, pursuer.y,
                pursuer.heading, self.obstacles,
            )
            self._last_lidar_readings.append(reading)

    def _build_observations(self):
        """构建每架追捕者的观测向量。"""
        target_state = self.target.get_state()
        obs = {}

        for i, pursuer in enumerate(self.pursuers):
            ps = pursuer.get_state()

            # 自身状态
            own_state = [ps["x"], ps["y"], ps["vx"], ps["vy"]]

            # 目标绝对状态
            target_abs = [
                target_state["x"], target_state["y"],
                target_state["vx"], target_state["vy"],
            ]

            # 目标相对极坐标
            dx = target_state["x"] - ps["x"]
            dy = target_state["y"] - ps["y"]
            dist = np.sqrt(dx ** 2 + dy ** 2)
            angle = np.arctan2(dy, dx) - ps["heading"]
            # 归一化角度到 [-pi, pi]
            angle = (angle + np.pi) % (2 * np.pi) - np.pi
            target_polar = [dist, angle]

            # LiDAR
            lidar = self._last_lidar_readings[i].tolist()

            # 队友位置
            teammates = []
            for j, other in enumerate(self.pursuers):
                if j != i:
                    teammates.extend([other.x, other.y])

            obs_vec = own_state + target_abs + target_polar + lidar + teammates
            obs[f"agent_{i}"] = np.array(obs_vec, dtype=np.float32)

        return obs

    def _check_collisions(self):
        """
        检查无人机-障碍物碰撞以及无人机间碰撞。

        Returns
        -------
        bool
            True 表示发生碰撞。
        """
        all_uavs = self.pursuers + [self.target]

        # 检查无人机 vs 障碍物
        for uav in all_uavs:
            for obs in self.obstacles:
                dist = np.linalg.norm(
                    np.array([uav.x, uav.y]) - obs.get_pos()
                )
                if dist < (self.r_safe + self.r_uav + obs.radius):
                    self.collision_occurred = True
                    return True

        # 检查无人机 vs 无人机
        for i in range(len(all_uavs)):
            for j in range(i + 1, len(all_uavs)):
                dist = np.linalg.norm(
                    np.array([all_uavs[i].x, all_uavs[i].y]) -
                    np.array([all_uavs[j].x, all_uavs[j].y])
                )
                if dist < (2 * self.r_uav):
                    self.collision_occurred = True
                    return True

        return False

    def _check_capture(self):
        """执行围捕成功判定。"""
        pursuer_states = [p.get_state() for p in self.pursuers]
        target_state = self.target.get_state()
        return self.capture_checker.check(pursuer_states, target_state)

    # ------------------------------------------------------------------
    # 奖励函数
    # ------------------------------------------------------------------

    def _compute_rewards(self, collision, captured):
        """
        计算每架追捕者的奖励。

        R = mu_near*R_near + mu_stage*R_stage + mu_finish*R_finish
            + mu_safe*R_safe + mu_time*R_time
        """
        rewards = {}

        for i, pursuer in enumerate(self.pursuers):
            r_near = self._reward_near(pursuer)
            r_stage = self._reward_stage(pursuer, i)
            r_finish = self._reward_finish(captured)
            r_safe = self._reward_safe(i)
            r_time = self.mu_time  # 每步固定

            # 碰撞额外惩罚
            r_collision_penalty = -50.0 if collision else 0.0

            total = (self.mu_near * r_near +
                     self.mu_stage * r_stage +
                     self.mu_finish * r_finish +
                     self.mu_safe * r_safe +
                     r_time +
                     r_collision_penalty)

            rewards[f"agent_{i}"] = total

        return rewards

    def _reward_near(self, pursuer):
        """
        目标靠近奖励。

        R_near = (v_i / v_max) * max(cos_theta_i, 0)
        cos_theta = dot(v_i, vector_to_target) / (|v_i| * |d_to_target|)
        """
        ps = pursuer.get_state()
        ts = self.target.get_state()

        v = np.array([ps["vx"], ps["vy"]])
        to_target = np.array([ts["x"] - ps["x"], ts["y"] - ps["y"]])

        speed = np.linalg.norm(v)
        dist = np.linalg.norm(to_target)

        if speed < 1e-6 or dist < 1e-6:
            return 0.0

        cos_theta = np.dot(v, to_target) / (speed * dist)
        return (speed / self.p_v_max) * max(cos_theta, 0.0)

    def _reward_stage(self, pursuer, agent_idx):
        """
        多阶段协作奖励。

        - 追踪阶段 (d > d_limit): 鼓励缩小包围圈 + 编队完整
        - 包围阶段 (d_limit > d > d_capture): 鼓励减小目标活动空间
        - 围捕阶段 (d <= d_limit & 角度好但 d > d_capture): 鼓励加速捕获
        """
        ps = pursuer.get_state()
        ts = self.target.get_state()
        dist = np.linalg.norm(
            np.array([ts["x"] - ps["x"], ts["y"] - ps["y"]])
        )

        # 计算所有追捕者到目标的平均距离
        all_dists = []
        for p in self.pursuers:
            p_pos = p.get_pos()
            t_pos = self.target.get_pos()
            all_dists.append(np.linalg.norm(p_pos - t_pos))
        d_mean = np.mean(all_dists)

        # 计算追捕者围绕目标的角度覆盖
        angles = []
        for p in self.pursuers:
            p_pos = p.get_pos()
            t_pos = self.target.get_pos()
            delta = p_pos - t_pos
            angles.append(np.arctan2(delta[1], delta[0]))
        angles_sorted = sorted(angles)
        max_gap = 0.0
        n = len(angles_sorted)
        for k in range(n):
            gap = angles_sorted[(k + 1) % n] - angles_sorted[k]
            if k == n - 1:
                gap += 2 * np.pi
            if gap > max_gap:
                max_gap = gap
        # 角度覆盖度
        angle_coverage = (2 * np.pi - max_gap) / (2 * np.pi)

        # 判断阶段
        if dist > self.d_limit or angle_coverage < 0.5:
            # ---- 追踪阶段 ----
            # 鼓励缩小平均距离 + 提高角度覆盖
            r_track = -0.001 * d_mean + 0.5 * angle_coverage
            return r_track

        elif dist > self.d_capture and angle_coverage < 0.83:
            # ---- 包围阶段 (300deg ~= 0.83) ----
            # 鼓励均匀分布 + 缩小距离
            r_encircle = angle_coverage - 0.002 * d_mean
            return r_encircle

        elif dist > self.d_capture:
            # ---- 围捕阶段: 距离 < d_limit, 角度好, 但距离还不够近 ----
            # 指数奖励鼓励加速
            d_sum = sum(all_dists)
            # 需要存储上一步的 d_sum
            if not hasattr(self, '_prev_d_sum'):
                self._prev_d_sum = d_sum
            r_capture = -np.exp(d_sum - self._prev_d_sum) + 1.0
            self._prev_d_sum = d_sum
            return r_capture

        else:
            return 0.0

    def _reward_finish(self, captured):
        """围捕任务完成时的大额奖励。"""
        return 100.0 if captured else 0.0

    def _reward_safe(self, agent_idx):
        """
        安全避障奖励。

        当 LiDAR 最小读数过小时惩罚: R_safe = -max(0, 1 - min(lidar))
        """
        lidar = self._last_lidar_readings[agent_idx]
        min_reading = np.min(lidar)
        return -max(0.0, 1.0 - min_reading)

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------

    def render(self):
        """
        简单终端渲染（打印当前状态摘要）。

        完整可视化请使用 scripts/visualize.py 的 matplotlib 版本。
        """
        ts = self.target.get_state()
        print(f"\n=== Step {self.current_step}/{self.max_step} ===")
        print(f"Target:  pos=({ts['x']:7.1f}, {ts['y']:7.1f})  "
              f"speed={ts['speed']:6.1f}  heading={np.degrees(ts['heading']):5.0f}deg")
        for i, p in enumerate(self.pursuers):
            ps = p.get_state()
            dist = np.linalg.norm(self.target.get_pos() - p.get_pos())
            print(f"  Pursuer_{i}: pos=({ps['x']:7.1f}, {ps['y']:7.1f})  "
                  f"speed={ps['speed']:6.1f}  d_to_target={dist:6.1f}")

    def close(self):
        pass
