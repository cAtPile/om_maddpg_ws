"""
质点运动学模型 — 无人机位置/速度更新、边界约束、限幅。

所有无人机（追捕者 + 目标）共用此模型。
"""

import numpy as np


class UAV:
    """
    二维质点运动学无人机模型。

    Parameters
    ----------
    uav_id : int or str
        无人机标识符。
    v_max : float
        最大速度标量 (m/s)。
    a_max : float
        最大加速度标量 (m/s²)。
    bounds : tuple[float, float, float, float]
        任务区域边界 (x_min, x_max, y_min, y_max)。
    """

    def __init__(self, uav_id, v_max, a_max, bounds):
        self.uav_id = uav_id
        self.v_max = v_max
        self.a_max = a_max
        self.x_min, self.x_max, self.y_min, self.y_max = bounds
        self.r_uav = 5.0  # 无人机半径 (m)，碰撞检测用

        # 状态变量
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.heading = 0.0  # rad
        self.ax = 0.0
        self.ay = 0.0

    def reset(self, x, y, heading=None, vx=0.0, vy=0.0):
        """
        重置无人机到指定状态。

        Parameters
        ----------
        x, y : float
            初始位置。
        heading : float or None
            初始航向角 (rad)。若为 None 则随机生成。
        vx, vy : float
            初始速度分量。
        """
        self.x = float(x)
        self.y = float(y)
        self.vx = float(vx)
        self.vy = float(vy)
        if heading is None:
            self.heading = np.random.uniform(0, 2 * np.pi)
        else:
            self.heading = float(heading)
        self.ax = 0.0
        self.ay = 0.0

    def step(self, ax, ay, dt):
        """
        执行一个仿真步长，更新位置和速度。

        Parameters
        ----------
        ax, ay : float
            加速度指令 (m/s²)。会被限幅到 [-a_max, a_max]。
        dt : float
            仿真时间步长 (s)。

        Returns
        -------
        dict
            更新后的状态信息，包含 x, y, vx, vy, speed, heading, ax, ay。
        """
        # 加速度限幅
        self.ax = np.clip(float(ax), -self.a_max, self.a_max)
        self.ay = np.clip(float(ay), -self.a_max, self.a_max)

        # 欧拉积分：更新速度
        self.vx += self.ax * dt
        self.vy += self.ay * dt

        # 速度限幅
        speed = np.sqrt(self.vx ** 2 + self.vy ** 2)
        if speed > self.v_max:
            self.vx = self.vx / speed * self.v_max
            self.vy = self.vy / speed * self.v_max
            speed = self.v_max

        # 欧拉积分：更新位置
        self.x += self.vx * dt
        self.y += self.vy * dt

        # 边界约束：反弹 + 能量损失
        self._apply_boundary_constraint()

        # 更新航向角
        if speed > 1e-6:
            self.heading = np.arctan2(self.vy, self.vx)
        # 若速度接近 0，保持原有航向

        return self.get_state()

    def _apply_boundary_constraint(self):
        """边界反弹，碰壁后速度反向并衰减 50%。"""
        if self.x < self.x_min:
            self.x = self.x_min
            self.vx = abs(self.vx) * 0.5
        elif self.x > self.x_max:
            self.x = self.x_max
            self.vx = -abs(self.vx) * 0.5

        if self.y < self.y_min:
            self.y = self.y_min
            self.vy = abs(self.vy) * 0.5
        elif self.y > self.y_max:
            self.y = self.y_max
            self.vy = -abs(self.vy) * 0.5

    def get_state(self):
        """返回当前完整状态 dict。"""
        return {
            "id": self.uav_id,
            "x": self.x,
            "y": self.y,
            "vx": self.vx,
            "vy": self.vy,
            "speed": np.sqrt(self.vx ** 2 + self.vy ** 2),
            "heading": self.heading,
            "ax": self.ax,
            "ay": self.ay,
        }

    def get_pos(self):
        """返回位置 tuple。"""
        return np.array([self.x, self.y])

    def get_vel(self):
        """返回速度 tuple。"""
        return np.array([self.vx, self.vy])
