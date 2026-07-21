# Phase 1: 2D 仿真环境 + 随机目标策略 — 技术路线

## 目标

构建论文第 4.2 节描述的无人机集群围捕 2D 仿真环境，接口兼容 Gymnasium，并实现**随机逃逸**目标策略，使环境可独立运行和调试。

---

## 1. 文件结构

```
convergence_verify/
├── env/
│   ├── __init__.py
│   ├── uav_env.py              # Gymnasium Env 主类
│   ├── kinematics.py           # 质点运动学模型
│   ├── obstacle.py             # 圆形障碍物
│   ├── lidar.py                # 16 通道激光雷达
│   ├── apollonius.py           # 阿波罗尼奥斯圆判定（简化版优先）
│   └── target_policy.py        # 目标逃逸策略（本阶段只做随机策略）
├── config/
│   └── config.yaml             # 参数集中管理
└── scripts/
    └── test_env.py             # 环境冒烟测试脚本
```

---

## 2. 模块详细设计

### 2.1 `kinematics.py` — 质点运动学

```
class UAV:
    x, y          # 位置 (m)
    vx, vy        # 速度 (m/s)
    heading       # 航向角 (rad)
    v_max         # 最大速度标量
    a_max         # 最大加速度标量

    def reset(pos, heading):
        # 重置位置、速度、航向

    def step(action_ax, action_ay, dt):
        # action ∈ [-a_max, a_max]
        # 1. 更新速度: v += a * dt
        # 2. 限幅速度: |v| = min(|v|, v_max)
        # 3. 更新位置: p += v * dt
        # 4. 更新航向: heading = atan2(vy, vx)
        # 5. 边界约束: 若越界则反弹 + 速度反向衰减
        # 返回: 新状态 dict
```

**边界约束逻辑**：
- 若 `x < -1000`：`x = -1000`, `vx = -vx * 0.5`（反弹 + 能量损失）
- 若 `x > 1000`：`x = 1000`, `vx = -vx * 0.5`
- y 同理

### 2.2 `obstacle.py` — 圆形障碍物

```
class Obstacle:
    x, y           # 圆心位置
    radius          # 半径 (100~150m)
    vx, vy         # 移动速度（静态时为 0）
    is_dynamic      # 是否动态

    def reset(x, y, radius, is_dynamic):
        # 初始化位置，随机初始速度方向

    def step(dt):
        # 若动态: 按恒定速度移动
        # 碰壁反弹 + 随机角度扰动 (±15°)
```

**碰撞检测**：
- 无人机与障碍物的距离 `l` 应满足：`l > R_safe + R_uav + r_obstacle`
- 每步检查所有无人机是否与任意障碍物碰撞

### 2.3 `lidar.py` — 16 通道激光雷达

```
class Lidar:
    n_channels = 16          # 通道数
    angles                   # [0, 22.5, 45, ..., 337.5] 度
    D_max                    # 最大探测距离

    def sense(uav_pos, uav_heading, obstacles):
        # 对 16 个方向分别发射射线
        # 返回每个方向上最近障碍物的归一化距离 [0, 1]
        # 1.0 = 无障碍物（或超出 D_max）
        # 0.0 = 紧贴障碍物
        # 若探测范围内无任何障碍物，返回 D_max 对应归一化值
```

**射线-圆相交计算**：
- 每条射线从无人机中心出发，沿 `heading + angle_k` 方向
- 对每个障碍物圆求交，取最近正交点距离
- 若射线被多个圆遮挡，取最小距离
- 距离 > D_max 时截断为 D_max

### 2.4 `apollonius.py` — 围捕判定（简化版）

**完整版逻辑（Phase 1 可直接实现）**：

```
def check_capture(pursuers, target):
    """
    对每架追捕者 i:
      1. 计算与目标的阿波罗尼奥斯圆:
         λ_i = v_pursuer / v_target
         center_O_i = (target_pos - λ_i² * pursuer_pos) / (1 - λ_i²)
         radius_i = λ_i * |pursuer_pos - target_pos| / |1 - λ_i²|
      2. 检查相邻圆是否相交:
         对相邻圆 i 和 j, 相交条件: |O_i - O_j| <= r_i + r_j
      3. 检查目标是否被包围:
         所有 d_i < d_capture (300m) 且 相邻圆两两相交
      4. 检查角度分布:
         追捕者围绕目标的角度跨度 > 300° (避免都在同一侧)
    返回: bool
    """
```

**初期简化判定（备选）**：
- 所有追捕者距目标 < d_capture（300m）
- 追捕者围绕目标的最大角度间隙 < 60°（即覆盖 > 300°）
- 此简化方案计算量小，方便快速调试

### 2.5 `target_policy.py` — 目标逃逸策略

#### 2.5.1 随机逃逸策略（本阶段实现）

```
class RandomEscapePolicy:
    def get_action(target, pursuers, obstacles):
        # 随机选择速度方向: θ ∈ [0, 2π)
        # 速度大小: v ∈ [0.5*v_max, v_max]
        # 转换为加速度输出（受限幅约束）
        # 返回: [ax, ay]
```

行为说明：
- 每个时间步完全随机选择运动方向和速度大小
- 不具备任何避障或躲避追捕者的智能
- 模拟完全无感知能力的无序逃逸
- 论文中作为最基础的基线测试场景

#### 2.5.2 APF 规则逃逸策略（Phase 1 后期或 Phase 2 实现）

```
class APFEscapePolicy:
    def get_action(target, pursuers, obstacles):
        # 1. 斥力: 每架追捕者产生斥力 f_rep_i = k_rep * v_i / d_i²
        #    方向: 从追捕者指向目标
        # 2. 引力: 安全区域边界产生引力 f_att = k_att * d_safe
        # 3. 合力 = Σf_rep + f_att
        # 4. 速度方向沿合力方向, 大小与合力大小相关
        # 返回: [ax, ay]
```

#### 2.5.3 智能逃逸策略（Phase 3 实现）

- 预训练一个 DDPG 策略网络
- 状态：自身位置速度 + 追捕者位置 + LiDAR
- 奖励：远离追捕者 + 存活时间 + 避障

### 2.6 `uav_env.py` — Gymnasium 环境主类

```
class UAVEnv(gym.Env):
    def __init__(config):
        # 加载 config.yaml 参数
        # 初始化: 3 pursuers + 1 target + 3 obstacles + 3 lidars
        # 定义 action_space, observation_space

    def reset(seed):
        # 目标固定在原点 [0, 0]
        # 3 架追捕者分布在半径 ~900m 圆上，角度间隔 ~120°
        # 障碍物随机初始化（但避免与无人机重叠）
        # 返回: observation dict

    def step(actions):
        # actions: shape (3, 2) — 3 架追捕者的加速度
        # 1. 目标根据策略生成动作并移动
        # 2. 追捕者执行动作并移动
        # 3. 障碍物步进
        # 4. 更新 LiDAR 读数
        # 5. 计算奖励（5 部分）
        # 6. 判定围捕成功 / 超时 / 碰撞
        # 返回: obs, reward, terminated, truncated, info

    def _compute_reward():
        # R_total = μ1*R_near + μ2*R_stage + μ3*R_finish + μ4*R_safe + μ5*R_time
```

**状态空间 (每架追捕者)**：
```
S_i = [
    x_i, y_i, vx_i, vy_i,                    # 自身状态 (4)
    x_target, y_target, vx_target, vy_target, # 目标真实状态 (4)
    d_target, angle_target,                    # 目标相对极坐标 (2)
    lidar_1, ..., lidar_16,                   # LiDAR 读数 (16)
    [x_j, y_j for j != i],                    # 队友位置 (2 * 2 = 4)
]
# 总计: 4 + 4 + 2 + 16 + 4 = 30 维
```

**动作空间**：
```
A_i = [ax, ay]     # 连续二维加速度, ∈ [-a_max, a_max]
```

**奖励计算**：

1. **R_near（目标靠近奖励）**：
   ```
   cos_θ = dot(v_i, vector_to_target) / (|v_i| * |d_to_target|)
   R_near = (v_i / v_max) * max(cos_θ, 0)
   ```
   鼓励追捕者以较大速度朝向目标移动。

2. **R_stage（多阶段协作奖励）**：
   - **追踪阶段**（d_i > d_limit=750m 或 S_r 较大）：
     ```
     R_track = -α * d_mean + β * (1/S_r)
     ```
     鼓励编队保持 + 缩小包围圈

   - **包围阶段**（d_limit > d_i > d_capture=300m 且包围未封闭）：
     ```
     R_encircle = -γ * S_target - δ * std(angles_between_pursuers)
     ```
     鼓励减小目标活动空间 + 均匀分布角度

   - **围捕阶段**（d_i < d_limit 且角度分布均匀但 d_i > d_capture）：
     ```
     R_capture = -exp(Σd_i - Σd_i_prev)
     ```
     鼓励加速缩小包围

3. **R_finish（围捕成功奖励）**：
   ```
   当 check_capture() == True 且所有 d_i < d_capture:
       R_finish = +100  （终局奖励）
       terminated = True
   ```

4. **R_safe（安全避障奖励）**：
   ```
   R_safe = -max(0, 1 - min(lidar_readings))
   ```
   16 通道 LiDAR 读数中最小值越小 → 惩罚越大

5. **R_time（时间惩罚）**：
   ```
   R_time = -0.1  （每步固定）
   ```

**终止条件**：
- `terminated`（True 终止）：围捕成功 / 无人机碰撞 / 目标逃出边界
- `truncated`（True 截断）：step >= maxStep (100)

---

## 3. 配置管理 `config.yaml`

```yaml
env:
  area_size: [2000, 2000]        # m
  area_bounds: [-1000, 1000, -1000, 1000]
  dt: 0.05                         # 仿真步长 (s)
  max_step: 100                    # 单回合最大步数

pursuer:
  num: 3
  v_max: 100                       # m/s
  a_max: 40                        # m/s²
  init_distance: 900               # 初始距目标的距离 (m)

target:
  v_max: 120                       # m/s
  a_max: 50                        # m/s²
  init_position: [0, 0]
  escape_policy: "random"          # random | apf | ddpg

obstacle:
  num: 3
  radius_range: [100, 150]         # m
  speed: 30                        # m/s (仅动态障碍物)
  dynamic: false                   # true = 动态, false = 静态

lidar:
  n_channels: 16
  d_max: 200                       # 最大探测距离 (m)

task:
  d_limit: 750                     # 包围距离阈值 (m)
  d_capture: 300                   # 捕获距离阈值 (m)
  r_safe: 10                       # 无人机与障碍物安全距离 (m)
  r_uav: 5                         # 无人机半径 (m)

reward:
  mu_near: 1.0
  mu_stage: 1.0
  mu_finish: 100.0
  mu_safe: 2.0
  mu_time: -0.1
```

---

## 4. 冒烟测试 `scripts/test_env.py`

```
1. 创建环境实例
2. reset() → 打印初始观测 shape、初始位置
3. 随机动作 step 10 次 → 打印每步的 obs, reward, done, info
4. 验证: 位置更新正确、边界约束生效、LiDAR 读数合法、碰撞检测触发
5. 可视化（可选）: matplotlib 绘制 x-y 平面轨迹
```

**过关标准**：
- 环境 reset → step × 100 不崩溃
- 所有无人机位置始终在边界内
- LiDAR 读数 ∈ [0, 1]
- 碰撞发生时正确触发 terminated
- 随机目标与追捕者轨迹可视化合理（无穿模）

---

## 5. 可视化工具 `scripts/visualize.py`（推荐但非必须）

使用 `matplotlib.animation` 实时绘制：
- 灰色矩形：任务区域边界
- 蓝色三角形：追捕者（带航向箭头）
- 红色五角星：目标（带航向箭头）
- 灰色圆形：障碍物（半透明）
- 蓝色虚线：阿波罗尼奥斯圆
- 绿色射线：LiDAR 扫描线

可逐帧保存为 GIF 或实时显示。

---

## 6. 本阶段完成标准

| 检查项 | 状态 |
|--------|------|
| `UAV` 运动学模型正确：加速→限幅→位移→边界反弹 | ☐ |
| `Obstacle` 静态/动态障碍物行为正确 | ☐ |
| `Lidar` 16 通道测距计算正确 | ☐ |
| `Apollonius` 围捕判定逻辑正确（简化版可接受） | ☐ |
| `RandomEscapePolicy` 目标随机逃逸可用 | ☐ |
| `UAVEnv` 兼容 `gym.Env` 接口 | ☐ |
| `test_env.py` 所有冒烟测试通过 | ☐ |
| 手动运行 10 个 episode，所有路径不出错 | ☐ |

---

## 7. 后续衔接

Phase 1 完成后，环境即可直接作为 Phase 2（LSTM-PINN 预训练）和 Phase 3（算法训练）的基础。后续阶段无需修改环境接口，只需：
- Phase 2：调用 `env.step()` 采集轨迹数据 → 训练 LSTM-PINN
- Phase 3：将 LSTM-PINN 预测输出接入 OM-MADDPG 的状态空间
