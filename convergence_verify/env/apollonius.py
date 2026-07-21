"""
阿波罗尼奥斯圆（Apollonius Circle）围捕判定。

多追捕者-单目标场景中，围捕成功的条件：
  1. 每架追捕者与目标的距离 < d_capture
  2. 相邻追捕者对应的阿波罗尼奥斯圆两两相交（封闭逃逸通道）
  3. 追捕者围绕目标的角度覆盖 > 300°（均匀包围）
"""

import numpy as np


class ApolloniusCapture:
    """
    基于阿波罗尼奥斯圆的多对一围捕成功判定器。

    Parameters
    ----------
    d_capture : float
        捕获距离阈值 (m)。追捕者距目标 < 此距离时可能触发判定。
    angle_coverage_threshold : float
        角度覆盖阈值 (度)，默认 300°。
        追捕者围绕目标的最大角度间隙必须 < 360° - 此阈值。
    """

    def __init__(self, d_capture=300.0, angle_coverage_threshold=300.0):
        self.d_capture = d_capture
        self.angle_gap_threshold = np.radians(360.0 - angle_coverage_threshold)

    def check(self, pursuer_states, target_state):
        """
        判定是否成功围捕。

        Parameters
        ----------
        pursuer_states : list[dict]
            所有追捕者状态，每项含 x, y, speed。
        target_state : dict
            目标状态，含 x, y, speed。

        Returns
        -------
        tuple[bool, dict]
            (is_captured, debug_info)
            debug_info 包含各项判定条件的结果。
        """
        n = len(pursuer_states)
        target_pos = np.array([target_state["x"], target_state["y"]])
        target_speed = target_state["speed"]

        # 条件 1：所有追捕者距目标 < d_capture
        distances = []
        for p in pursuer_states:
            d = np.linalg.norm(np.array([p["x"], p["y"]]) - target_pos)
            distances.append(d)

        all_within_capture = all(d < self.d_capture for d in distances)

        # 条件 2：阿波罗尼奥斯圆两两相交
        circles = []
        for p in pursuer_states:
            center, radius = self._compute_apollonius_circle(
                np.array([p["x"], p["y"]]),
                p["speed"],
                target_pos,
                target_speed,
            )
            circles.append({"center": center, "radius": radius})

        all_adjacent_intersect = self._check_adjacent_circles_intersect(circles)

        # 条件 3：角度覆盖
        angles = []
        for p in pursuer_states:
            delta = np.array([p["x"], p["y"]]) - target_pos
            angle = np.arctan2(delta[1], delta[0])
            angles.append(angle)
        angles_sorted = sorted(angles)

        max_gap = 0.0
        for i in range(n):
            gap = angles_sorted[(i + 1) % n] - angles_sorted[i]
            if i == n - 1:
                gap += 2 * np.pi  # 跨过 0 度线
            if gap > max_gap:
                max_gap = gap

        angle_covered = max_gap < self.angle_gap_threshold

        # 综合判定
        captured = all_within_capture and all_adjacent_intersect and angle_covered

        debug_info = {
            "distances": distances,
            "all_within_capture": all_within_capture,
            "circles": circles,
            "all_adjacent_intersect": all_adjacent_intersect,
            "max_angle_gap_deg": np.degrees(max_gap),
            "angle_covered": angle_covered,
            "captured": captured,
        }

        return captured, debug_info

    def _compute_apollonius_circle(self, pursuer_pos, pursuer_speed, target_pos, target_speed):
        """
        计算追捕者与目标之间的阿波罗尼奥斯圆。

        由 |P - M| / |E - M| = λ 定义的所有点 M 组成的圆，
        其中 λ = v_p / v_e < 1。

        Returns
        -------
        tuple[np.ndarray, float]
            (圆心坐标, 半径)
        """
        if target_speed < 1e-6:
            # 目标静止：追捕者总比目标快到达任意点
            # 返回一个以追捕者为中心的大圆
            center = pursuer_pos.copy()
            radius = 1000.0
            return center, radius

        lam = pursuer_speed / target_speed

        if abs(lam - 1.0) < 1e-6:
            # λ = 1：退化情况，阿波罗尼奥斯圆退化为垂直平分线
            # 返回一个很大的圆近似
            mid = (pursuer_pos + target_pos) / 2.0
            radius = 2000.0
            return mid, radius

        # 圆心：O = (E - λ²*P) / (1 - λ²)
        # 设 P = pursuer, E = target
        lam2 = lam ** 2
        center = (target_pos - lam2 * pursuer_pos) / (1.0 - lam2)

        # 半径：r = λ * |P - E| / |1 - λ²|
        d_pe = np.linalg.norm(pursuer_pos - target_pos)
        radius = lam * d_pe / abs(1.0 - lam2)

        return center, radius

    def _check_adjacent_circles_intersect(self, circles):
        """
        检查相邻（按圆心的角度排序后）的阿波罗尼奥斯圆是否两两相交。

        相邻圆相交判定：|O_i - O_j| <= r_i + r_j
        """
        n = len(circles)
        if n <= 1:
            return False

        # 按圆心相对几何中心的极角排序，定义相邻关系
        geo_center = np.mean([c["center"] for c in circles], axis=0)
        angles = []
        for c in circles:
            delta = c["center"] - geo_center
            angles.append(np.arctan2(delta[1], delta[0]))

        sorted_indices = np.argsort(angles)

        for k in range(n):
            i = sorted_indices[k]
            j = sorted_indices[(k + 1) % n]
            c_i = circles[i]
            c_j = circles[j]
            dist = np.linalg.norm(c_i["center"] - c_j["center"])
            if dist > c_i["radius"] + c_j["radius"]:
                return False

        return True
