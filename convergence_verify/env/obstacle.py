"""
圆形障碍物模型 — 支持静态和动态两种模式。

动态障碍物以恒定速度直线运动，碰壁后反弹并随机调整方向。
"""

import numpy as np


class Obstacle:
    """
    圆形障碍物。

    Parameters
    ----------
    obstacle_id : int
        障碍物标识符。
    radius : float
        半径 (m)。
    speed : float
        移动速度 (m/s)，仅动态模式使用。
    is_dynamic : bool
        是否为动态障碍物。
    bounds : tuple[float, float, float, float]
        任务区域边界 (x_min, x_max, y_min, y_max)。
    """

    def __init__(self, obstacle_id, radius, speed, is_dynamic, bounds):
        self.obstacle_id = obstacle_id
        self.radius = float(radius)
        self.speed = float(speed)
        self.is_dynamic = is_dynamic
        self.x_min, self.x_max, self.y_min, self.y_max = bounds

        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0

    def reset(self, x, y):
        """
        重置障碍物位置，动态障碍物随机初始化速度方向。

        Parameters
        ----------
        x, y : float
            初始圆心位置。
        """
        self.x = float(x)
        self.y = float(y)

        if self.is_dynamic:
            angle = np.random.uniform(0, 2 * np.pi)
            self.vx = self.speed * np.cos(angle)
            self.vy = self.speed * np.sin(angle)
        else:
            self.vx = 0.0
            self.vy = 0.0

    def step(self, dt):
        """
        障碍物步进。动态障碍物匀速移动，碰壁反弹并加入随机方向扰动。

        Parameters
        ----------
        dt : float
            仿真时间步长。
        """
        if not self.is_dynamic:
            return

        # 移动
        self.x += self.vx * dt
        self.y += self.vy * dt

        # 边界反弹 + 随机角度扰动
        angle_change = np.random.uniform(-np.pi / 12, np.pi / 12)  # ±15°

        if self.x - self.radius < self.x_min:
            self.x = self.x_min + self.radius
            self.vx = abs(self.vx)
            angle = np.arctan2(self.vy, self.vx) + angle_change
            self.vx = self.speed * np.cos(angle)
            self.vy = self.speed * np.sin(angle)

        elif self.x + self.radius > self.x_max:
            self.x = self.x_max - self.radius
            self.vx = -abs(self.vx)
            angle = np.arctan2(self.vy, self.vx) + angle_change
            self.vx = self.speed * np.cos(angle)
            self.vy = self.speed * np.sin(angle)

        if self.y - self.radius < self.y_min:
            self.y = self.y_min + self.radius
            self.vy = abs(self.vy)
            angle = np.arctan2(self.vy, self.vx) + angle_change
            self.vx = self.speed * np.cos(angle)
            self.vy = self.speed * np.sin(angle)

        elif self.y + self.radius > self.y_max:
            self.y = self.y_max - self.radius
            self.vy = -abs(self.vy)
            angle = np.arctan2(self.vy, self.vx) + angle_change
            self.vx = self.speed * np.cos(angle)
            self.vy = self.speed * np.sin(angle)

    def get_pos(self):
        """返回圆心位置。"""
        return np.array([self.x, self.y])

    def get_state(self):
        """返回障碍物当前状态。"""
        return {
            "id": self.obstacle_id,
            "x": self.x,
            "y": self.y,
            "radius": self.radius,
            "vx": self.vx,
            "vy": self.vy,
            "is_dynamic": self.is_dynamic,
        }

    def contains_point(self, px, py):
        """检查点 (px, py) 是否在障碍物内。"""
        dist = np.sqrt((px - self.x) ** 2 + (py - self.y) ** 2)
        return dist < self.radius
