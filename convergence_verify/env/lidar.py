"""
16 通道激光雷达模拟。

均匀覆盖 360° 方位，每条射线通过射线-圆相交计算障碍物距离。
"""

import numpy as np


class Lidar:
    """
    16 通道 2D 激光雷达模拟器。

    Parameters
    ----------
    n_channels : int
        激光通道数，默认 16。
    d_max : float
        最大探测距离 (m)。
    """

    def __init__(self, n_channels=16, d_max=200.0):
        self.n_channels = n_channels
        self.d_max = float(d_max)

        # 16 个通道的固定角度（弧度），以机头方向为 0°
        # 通道 0 在正前方 (0°)，按逆时针排列
        self.channel_angles = np.linspace(0.0, 2 * np.pi, n_channels, endpoint=False)

    def sense(self, uav_x, uav_y, uav_heading, obstacles):
        """
        以无人机当前位置和航向，对每个激光通道测距。

        Parameters
        ----------
        uav_x, uav_y : float
            无人机位置。
        uav_heading : float
            无人机航向角 (rad)。
        obstacles : list[Obstacle]
            环境中所有障碍物。

        Returns
        -------
        np.ndarray
            shape (n_channels,)，归一化距离 [0, 1]。
            1.0 = 探测范围内无任何障碍物。
            0.0 = 障碍物在 0 距离（紧贴）。
        """
        readings = np.ones(self.n_channels) * self.d_max

        for i, rel_angle in enumerate(self.channel_angles):
            # 绝对射线方向角
            ray_angle = uav_heading + rel_angle
            ray_dir = np.array([np.cos(ray_angle), np.sin(ray_angle)])
            uav_pos = np.array([uav_x, uav_y])

            min_dist = self.d_max
            for obs in obstacles:
                dist = self._ray_circle_intersection(uav_pos, ray_dir, obs.get_pos(), obs.radius)
                if dist is not None and dist < min_dist:
                    min_dist = dist

            readings[i] = min_dist

        # 截断到 d_max 并归一化到 [0, 1]
        readings = np.clip(readings, 0.0, self.d_max)
        normalized = readings / self.d_max
        return normalized

    def _ray_circle_intersection(self, ray_origin, ray_dir, circle_center, circle_radius):
        """
        计算射线与圆的最近交点距离。

        射线方程: P = ray_origin + t * ray_dir, t >= 0
        圆方程:   |P - circle_center|² = r²

        Returns
        -------
        float or None
            最近正交点距离；若不相交返回 None。
        """
        oc = ray_origin - circle_center
        a = np.dot(ray_dir, ray_dir)  # = 1.0，因为 ray_dir 是单位向量
        b = 2.0 * np.dot(oc, ray_dir)
        c = np.dot(oc, oc) - circle_radius ** 2

        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return None  # 不相交

        sqrt_d = np.sqrt(discriminant)
        t1 = (-b - sqrt_d) / (2 * a)
        t2 = (-b + sqrt_d) / (2 * a)

        # 取最小的正 t
        if t1 >= 0:
            return t1
        elif t2 >= 0:
            return t2
        else:
            return None  # 交点在射线后方
