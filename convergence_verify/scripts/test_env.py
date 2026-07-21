#!/usr/bin/env python3
"""
环境冒烟测试脚本。

验证:
  1. reset() → 观测 shape 正确
  2. 随机动作 step 100 步不崩溃
  3. 位置更新、边界约束生效
  4. LiDAR 读数合法
  5. 碰撞检测触发
"""

import sys
import os

# 将 convergence_verify 加入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from env.uav_env import UAVEnv


def test_env_basic():
    """基础测试：reset + step × max_step 不崩溃。"""
    print("=" * 60)
    print("Test 1: 基础 reset + step 测试")
    print("=" * 60)

    env = UAVEnv()
    obs, info = env.reset(seed=42)

    # 检查观测结构
    print(f"\n观测键: {list(obs.keys())}")
    for k, v in obs.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}, "
              f"range=[{v.min():.3f}, {v.max():.3f}]")

    print(f"\n初始目标位置: {info['target_pos']}")
    print(f"初始追捕者位置:")
    for i, pos in enumerate(info['pursuer_positions']):
        print(f"  Pursuer_{i}: {pos}")

    # 运行一个完整 episode
    total_reward = 0.0
    for step in range(env.max_step):
        # 随机动作
        actions = np.random.uniform(-1, 1, (env.n_pursuers, 2))
        obs, rewards, terminated, truncated, info = env.step(actions)

        # 聚合奖励
        step_reward = sum(rewards.values())
        total_reward += step_reward

        if step < 3 or step % 20 == 0:
            print(f"\nStep {step+1}: total_reward={step_reward:.3f}, "
                  f"captured={info['captured']}, collision={info['collision']}")
            ts = info['target_state']
            print(f"  Target: ({ts['x']:.1f}, {ts['y']:.1f}), speed={ts['speed']:.1f}")

        if terminated:
            print(f"\n>>> Episode 在 step {step+1} 终止: "
                  f"captured={info['captured']}, collision={info['collision']}")
            break
        if truncated:
            print(f"\n>>> Episode 在 step {step+1} 超时截断")
            break

    print(f"\n总奖励: {total_reward:.2f}")
    print("基础测试通过 ✓")
    env.close()


def test_boundary_constraint():
    """测试边界约束是否生效。"""
    print("\n" + "=" * 60)
    print("Test 2: 边界约束测试")
    print("=" * 60)

    env = UAVEnv()
    env.reset(seed=123)

    # 强制所有追捕者向边界外加速，验证会不会被约束
    out_of_bounds_detected = False

    for step in range(50):
        # 向 (1000, 1000) 方向加速——推向东北角
        for i, p in enumerate(env.pursuers):
            to_corner = np.array([1000.0 - p.x, 1000.0 - p.y])
            dist = np.linalg.norm(to_corner)
            if dist > 1e-6:
                direction = to_corner / dist
            else:
                direction = np.array([0.0, 0.0])
            actions = direction.reshape(1, 2).repeat(env.n_pursuers, axis=0)

        obs, rewards, terminated, truncated, info = env.step(actions)

        for p in env.pursuers:
            if p.x > env.x_max or p.x < env.x_min:
                out_of_bounds_detected = True
                print(f"  FAIL: Pursuer {p.uav_id} x={p.x:.1f} 超出边界!")
            if p.y > env.y_max or p.y < env.y_min:
                out_of_bounds_detected = True
                print(f"  FAIL: Pursuer {p.uav_id} y={p.y:.1f} 超出边界!")

    if not out_of_bounds_detected:
        print("所有无人机位置始终在边界内 ✓")
    else:
        print("边界约束测试失败 ✗")

    env.close()


def test_lidar_readings():
    """验证 LiDAR 读数合法性。"""
    print("\n" + "=" * 60)
    print("Test 3: LiDAR 读数测试")
    print("=" * 60)

    env = UAVEnv()
    env.reset(seed=456)

    all_valid = True

    for step in range(30):
        actions = np.random.uniform(-1, 1, (env.n_pursuers, 2))
        obs, rewards, terminated, truncated, info = env.step(actions)

        for i, readings in enumerate(info['lidar_readings']):
            if readings.min() < 0.0 or readings.max() > 1.0:
                print(f"  FAIL Step {step}: Pursuer_{i} LiDAR out of [0,1]: "
                      f"min={readings.min():.4f} max={readings.max():.4f}")
                all_valid = False

        if terminated:
            break

    if all_valid:
        print("所有 LiDAR 读数 ∈ [0, 1] ✓")
    else:
        print("LiDAR 读数测试失败 ✗")

    env.close()


def test_collision_detection():
    """验证碰撞检测。"""
    print("\n" + "=" * 60)
    print("Test 4: 碰撞检测测试")
    print("=" * 60)

    env = UAVEnv()
    env.reset(seed=789)

    # 手动把一架追捕者放到障碍物中心，期望检测到碰撞
    if len(env.obstacles) > 0:
        obs0 = env.obstacles[0]
        env.pursuers[0].x = obs0.x
        env.pursuers[0].y = obs0.y
        env.pursuers[0].vx = 0.0
        env.pursuers[0].vy = 0.0

    # 走一步，碰撞检测应触发
    actions = np.zeros((env.n_pursuers, 2))
    obs, rewards, terminated, truncated, info = env.step(actions)

    if info['collision']:
        print(f"碰撞检测正确触发: collision=True ✓")
    else:
        print(f"WARNING: 追捕者在障碍物中心但未触发碰撞 (可能边界反弹改变了坐标)")
        # 多试几步
        collision_triggered = False
        for _ in range(10):
            env.pursuers[0].x = env.obstacles[0].x
            env.pursuers[0].y = env.obstacles[0].y
            actions = np.zeros((env.n_pursuers, 2))
            obs, rewards, terminated, truncated, info = env.step(actions)
            if info['collision']:
                collision_triggered = True
                break
        if collision_triggered:
            print("碰撞检测在重试后触发 ✓")
        else:
            print("碰撞检测未触发 — 请检查逻辑 ✗")

    env.close()


def test_reproducibility():
    """验证相同 seed 产生相同初始状态。"""
    print("\n" + "=" * 60)
    print("Test 5: 可复现性测试")
    print("=" * 60)

    env1 = UAVEnv()
    env2 = UAVEnv()

    obs1, info1 = env1.reset(seed=42)
    obs2, info2 = env2.reset(seed=42)

    # 比较目标位置
    match = np.allclose(info1['target_pos'], info2['target_pos'], atol=1e-6)
    if match:
        print("相同 seed 产生相同初始状态 ✓")
    else:
        print(f"状态不一致: {info1['target_pos']} vs {info2['target_pos']} ✗")

    env1.close()
    env2.close()


if __name__ == "__main__":
    test_env_basic()
    test_boundary_constraint()
    test_lidar_readings()
    test_collision_detection()
    test_reproducibility()

    print("\n" + "=" * 60)
    print("全部冒烟测试完成")
    print("=" * 60)
