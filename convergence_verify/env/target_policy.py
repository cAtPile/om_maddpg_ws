"""
目标无人机逃逸策略。

实现论文中描述的三类逃逸策略：
  - RandomEscapePolicy:  随机逃逸
  - APFEscapePolicy:     基于人工势场法 (APF) 的规则逃逸
  - DDPGEscapePolicy:    智能逃逸（Phase 3 实现，本文件保留接口）
"""

import numpy as np


class RandomEscapePolicy:
    """
    随机逃逸策略。

    每个时间步随机选择运动方向和速度大小，模拟完全无感知能力的无序逃逸。
    对应论文公式 (4.21): v_t = v_rand。
    """

    def __init__(self, v_max=120.0, a_max=50.0):
        self.v_max = v_max
        self.a_max = a_max

    def get_action(self, target, pursuers, obstacles, lidar_readings=None):
        """
        生成随机逃逸加速度。

        Parameters
        ----------
        target : UAV
            目标无人机（含当前状态）。
        pursuers : list[UAV]
            追捕者列表（本策略不使用）。
        obstacles : list[Obstacle]
            障碍物列表（本策略不使用）。
        lidar_readings : np.ndarray or None
            LiDAR 读数（本策略不使用）。

        Returns
        -------
        tuple[float, float]
            (ax, ay) 加速度指令。
        """
        ax = np.random.uniform(-self.a_max, self.a_max)
        ay = np.random.uniform(-self.a_max, self.a_max)
        return ax, ay

    def get_target_velocity(self, target, pursuers, obstacles):
        """
        生成随机目标速度（用于直接速度控制）。

        Returns
        -------
        np.ndarray
            [vx, vy] 速度向量。
        """
        angle = np.random.uniform(0, 2 * np.pi)
        speed = np.random.uniform(0.3 * self.v_max, self.v_max)
        return np.array([speed * np.cos(angle), speed * np.sin(angle)])


class APFEscapePolicy:
    """
    基于人工势场法（APF）的规则逃逸策略。

    追捕者产生斥力 (f_rep)，安全区域边界产生引力 (f_att)。
    合力决定目标运动方向。对应论文公式 (4.22)-(4.25)。

    Parameters
    ----------
    v_max : float
        目标最大速度 (m/s)。
    a_max : float
        目标最大加速度 (m/s²)。
    k_rep : float
        追捕斥力系数。
    k_att : float
        安全引力系数。
    bounds : tuple
        任务区域边界 (x_min, x_max, y_min, y_max)。
    """

    def __init__(self, v_max=120.0, a_max=50.0, k_rep=500.0, k_att=10.0,
                 bounds=(-1000, 1000, -1000, 1000)):
        self.v_max = v_max
        self.a_max = a_max
        self.k_rep = k_rep
        self.k_att = k_att
        self.x_min, self.x_max, self.y_min, self.y_max = bounds

    def get_action(self, target, pursuers, obstacles, lidar_readings=None):
        """
        基于 APF 合力计算加速度。

        Parameters
        ----------
        target : UAV
            目标无人机。
        pursuers : list[UAV]
            所有追捕者。
        obstacles : list[Obstacle]
            所有障碍物。
        lidar_readings : np.ndarray or None
            LiDAR 读数。

        Returns
        -------
        tuple[float, float]
            (ax, ay) 加速度指令。
        """
        target_pos = target.get_pos()
        force_total = np.zeros(2)

        # ---- 斥力：每架追捕者对目标产生斥力 ----
        for pursuer in pursuers:
            p_pos = pursuer.get_pos()
            delta = target_pos - p_pos  # 从追捕者指向目标
            dist = np.linalg.norm(delta)
            if dist < 1e-3:
                dist = 1e-3

            # 斥力大小: f_rep_i = k_rep * v_i / d_i^2  (论文公式 4.23)
            rep_magnitude = self.k_rep * pursuer.get_state()["speed"] / (dist ** 2)
            force_total += rep_magnitude * (delta / dist)

        # ---- 引力：区域边界产生引力 ----
        # 找到最近的边界，向远离该边界的中心方向产生引力
        # 论文公式 (4.24): f_att = k_att * d_safe
        center = np.array([(self.x_min + self.x_max) / 2.0,
                           (self.y_min + self.y_max) / 2.0])

        # 目标希望远离追捕者，趋向远离追捕者集群方向的区域边缘
        # 简化：如果目标距边界太近，向中心产生引力
        margin = 200.0  # 安全边距
        to_center = center - target_pos
        dist_to_center = np.linalg.norm(to_center)

        # 边界排斥：距边界越近引力越大
        x_margin = min(abs(target_pos[0] - self.x_min),
                       abs(target_pos[0] - self.x_max))
        y_margin = min(abs(target_pos[1] - self.y_min),
                       abs(target_pos[1] - self.y_max))
        min_margin = min(x_margin, y_margin)

        if min_margin < margin:
            # 边距小：向中心区域引导
            att_magnitude = self.k_att * (margin - min_margin) / margin
            if dist_to_center > 1e-3:
                force_total += att_magnitude * (to_center / dist_to_center)

        # ---- 避障力：基于 LiDAR 读数的局部避障 ----
        if lidar_readings is not None:
            # 找到最近的障碍物方向
            min_idx = np.argmin(lidar_readings)
            min_dist = lidar_readings[min_idx]
            if min_dist < 0.3:  # 障碍物较近
                # 按 16 通道的角度计算对应方向
                angle = target.heading + 2 * np.pi * min_idx / len(lidar_readings)
                # 向相反方向产生规避力
                avoid_dir = np.array([-np.cos(angle), -np.sin(angle)])
                avoid_magnitude = (0.3 - min_dist) / 0.3 * self.a_max * 5.0
                force_total += avoid_magnitude * avoid_dir

        # ---- 合力 -> 期望速度 -> 加速度 ----
        force_magnitude = np.linalg.norm(force_total)
        if force_magnitude > 1e-6:
            # 合力方向 = 期望速度方向
            desired_direction = force_total / force_magnitude
            # 期望速度大小与合力大小相关，但不超过 v_max
            desired_speed = min(force_magnitude / (self.k_rep * 3.0), self.v_max)
            desired_speed = max(desired_speed, 0.3 * self.v_max)
        else:
            desired_direction = np.array([1.0, 0.0])
            desired_speed = 0.3 * self.v_max

        desired_vel = desired_speed * desired_direction
        current_vel = target.get_vel()

        # 加速度 = (期望速度 - 当前速度) 按比例
        accel = (desired_vel - current_vel) * 2.0  # 比例系数

        # 限幅
        accel_magnitude = np.linalg.norm(accel)
        if accel_magnitude > self.a_max:
            accel = accel / accel_magnitude * self.a_max

        return accel[0], accel[1]


class DDPGEscapePolicy:
    """
    智能逃逸策略（占位接口）。

    预训练 DDPG 策略网络，具备避障和逃脱能力。
    Phase 3 实现。
    """

    def __init__(self, v_max=120.0, a_max=50.0):
        self.v_max = v_max
        self.a_max = a_max

    def get_action(self, target, pursuers, obstacles, lidar_readings=None):
        """
        DDPG 策略生成加速度（待实现）。

        Returns
        -------
        tuple[float, float]
            (ax, ay) 加速度指令。
        """
        # Phase 3: 加载预训练的 DDPG 网络，根据状态输出动作
        raise NotImplementedError("DDPG escape policy will be implemented in Phase 3")
