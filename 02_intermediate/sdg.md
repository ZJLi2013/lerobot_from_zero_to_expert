# LeRobot 合成数据生成（SDG）技术指南

> 目标：理解 LeRobot 数据结构的内在含义，并在此基础上构建合成数据管线

---

## 目录

1. [LeRobot 数据集字段精解](#1-lerobot-数据集字段精解)
2. [核心问题：state / action 从哪里来](#2-核心问题state--action-从哪里来)
3. [从视频反推 state/action 的可行性](#3-从视频反推-stateaction-的可行性)
4. [SDG 技术路径总览](#4-sdg-技术路径总览)
5. [路径 A：物理仿真采集（最完整）](#5-路径-a物理仿真采集最完整)
6. [路径 B：视频运动重建（Video → State）](#6-路径-b视频运动重建video--state)
7. [路径 C：World Model 生成新观测](#7-路径-c-world-model-生成新观测)
8. [路径 D：数据增强（最快上手）](#8-路径-d数据增强最快上手)
9. [写入 LeRobot 格式的 API](#9-写入-lerobot-格式的-api)
10. [推荐起步方案](#10-推荐起步方案)
11. [Genesis 仿真引擎可行性评估](#11-genesis-仿真引擎可行性评估)

---

## 1. LeRobot 数据集字段精解

### 1.1 数据集目录结构

```
dataset_root/
├── meta/
│   ├── info.json          # 元信息：特征定义、fps、总帧数、机器人类型
│   ├── stats.json         # 每个特征的统计信息（min/max/mean/std/分位数）
│   ├── tasks.parquet      # 任务描述（episode_index → task 字符串）
│   └── episodes/          # episode 级别的元数据（from/to index, task_index）
├── data/
│   └── chunk-000/
│       └── file-000.parquet  # 表格数据（state, action, timestamp, index等）
└── videos/
    ├── observation.images.up/
    │   └── chunk-000/episode_000000.mp4
    └── observation.images.side/
        └── chunk-000/episode_000000.mp4
```

### 1.2 字段详解：svla_so101_pickplace 为例

#### `observation.state`（6 维，float32）

```
[ shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper ]
```

| 属性 | 说明 |
|------|------|
| **来源** | SO-101 Dynamixel 编码器读数，通过 `bus.sync_read("Present_Position")` 获取 |
| **物理含义** | **当前关节位置**（位置控制，不是速度或力矩） |
| **单位** | **角度（度，degrees）**，以关节行程中点为 0，范围约 [-180, 180] |
| **归一化** | `(raw_val - mid) × 360 / max_resolution`，其中 mid = (range_min + range_max) / 2 |
| **与弧度关系** | 训练数据中为度；部分第三方数据集（如 phospho）用弧度，需注意换算 |
| **维度 6** | 5 个旋转关节（肩/肘/腕×2/夹爪旋转）+ 1 个夹爪开合 |

#### `action`（6 维，float32）

```
[ shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos, gripper.pos ]
```

| 属性 | 说明 |
|------|------|
| **来源** | Teleop 时：Leader arm 的关节位置（通过 `so_leader.get_action()` 获取） |
| **物理含义** | **目标关节位置**（发给 Follower arm 的 `Goal_Position`） |
| **单位** | 与 `observation.state` 完全一致（度） |
| **是否增量** | **绝对值**，不是相对当前状态的增量 |
| **采集延迟** | action[t] 对应的观测通常是 state[t] 或 state[t-1]，取决于控制循环 |

> **关键理解**：`state` = "机器人现在在哪里"，`action` = "机器人下一步要去哪"。两者单位相同，值域相近，但 action 是 leader arm 的位置，state 是 follower arm 的位置，两者之间的差即为跟踪误差。

#### `observation.images.up` / `observation.images.side`

| 属性 | 说明 |
|------|------|
| **形状** | `(C, H, W)` = `(3, H, W)`，float32，值域 [0, 1] |
| **存储** | 压缩为 MP4 视频（libsvtav1 编解码），按 episode 存放 |
| **加载** | PyAV / torchcodec 按需解码 |
| **时间对齐** | 与 state/action 同步，同一 timestamp 下的图像和数值 |

#### 时序与索引字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `timestamp` | float32 | 帧的绝对时间戳（秒），episode 内从 0 开始 |
| `frame_index` | int64 | episode 内的帧序号（0-based）|
| `episode_index` | int64 | episode 编号（全局）|
| `index` | int64 | 全局帧编号（跨所有 episode 唯一）|
| `task_index` | int64 | 对应 tasks.parquet 中的任务行 |
| `task` | str | 自然语言任务描述，**每帧都有**（不是每 episode 一个）|

#### `next.done`（可选）

仅在 RL 场景使用，标记 episode 终止。`lerobot_record.py` 录制时**不写入**；RL Buffer 在训练时根据 episode 边界自动推断。

### 1.3 stats.json 结构

```json
{
  "observation.state": {
    "min":  [-90.2, -120.1, -10.3, -45.0, -180.0, -5.0],
    "max":  [ 90.1,   30.5, 125.8,  45.2,  180.0, 95.0],
    "mean": [ 18.7,  -57.8,  67.7,  69.7,  -48.3,  6.9],
    "std":  [ 26.0,   40.3,  28.9,  11.3,    7.1,  7.5],
    "count": 11939
  },
  "action": { ... },
  "observation.images.up": { "mean": [0.485, 0.456, 0.406], ... }
}
```

stats 用于策略模型的归一化层（`NormalizationMode.MEAN_STD`）。

---

## 2. 核心问题：state / action 从哪里来

```
┌─────────────────────────────────────────────────────────────────┐
│                    数据采集方式对比                               │
│                                                                 │
│  ① 真实机器人遥操作（当前 SVLA 数据集的来源）                      │
│     Leader arm → encoder → action                               │
│     Follower arm → encoder → state                              │
│     Camera → 图像                                                │
│                                                                 │
│  ② 物理仿真（MuJoCo / Isaac Sim）                                │
│     仿真环境直接提供 state + action，无需真实硬件                   │
│     图像通过渲染器生成（可真实感渲染）                              │
│                                                                 │
│  ③ 视频运动重建（逆问题，最难）                                    │
│     只有视频 → 估算 3D 位姿 → IK → 关节角度                        │
│     需要标定相机 + 已知 URDF + 准确的姿态估计                       │
│     精度受限，适合粗粒度数据                                        │
│                                                                 │
│  ④ World Model（生成）                                           │
│     给定 state/action → 生成新的视觉观测                           │
│     反过来：给定视频 → 学习隐式 state 表示                          │
└─────────────────────────────────────────────────────────────────┘
```

**根本原因**：`observation.state` 和 `action` 都是**编码器读数**（proprioceptive），不是视觉信息。摄像头看不到关节角，只能看到末端位置和外观。这是机器人数据采集的本质约束。

---

## 3. 从视频反推 state/action 的可行性

### 3.1 问题定义

给定一段 SO-101 抓取视频（单目/双目），要求恢复每帧的 `state`（6D 关节角）。

### 3.2 技术管线

```
视频帧
  │
  ▼ [目标检测 / 分割]
机器人各连杆像素坐标
  │
  ▼ [2D → 3D 提升]
各连杆 3D 端点坐标（需要相机内参 + 外参）
  │
  ▼ [逆运动学 IK]  ← 需要 SO-101 URDF
各关节角度（degrees）
  │
  ▼
observation.state / action
```

### 3.3 lerobot 内置的 IK 支持

lerobot 有完整的 FK/IK 工具链（基于 [placo](https://github.com/Rhoban/placo) 库）：

```python
# src/lerobot/model/kinematics.py
from lerobot.model.kinematics import RobotKinematics

# 初始化（需要 SO-101 URDF）
kin = RobotKinematics(
    urdf_path="path/to/so101_new_calib.urdf",
    target_frame_name="gripper_frame_link",
)

# 正向运动学：关节角 → 末端位姿
joint_deg = np.array([18.7, -57.8, 67.7, 69.7, -48.3, 6.9])
ee_pose_4x4 = kin.forward_kinematics(joint_deg)  # 4×4 齐次变换矩阵

# 逆运动学：末端目标位姿 → 关节角
target_T = ee_pose_4x4  # 目标末端位姿
joint_solution = kin.inverse_kinematics(
    target_T,
    initial_joint_pos=joint_deg  # 初始猜测
)
```

### 3.4 当前可行性评估

| 条件 | 是否满足 | 说明 |
|------|---------|------|
| IK 工具 | ✓ | lerobot 内置，基于 placo |
| SO-101 URDF | ✓ | so-arm100 仓库提供 |
| 3D 位姿估计 | ✗ | **lerobot 无内置工具** |
| 相机标定 | 视情况 | svla 数据集未公开标定参数 |

**结论**：从视频反推关节角是可行的工程问题，但 lerobot 未提供开箱即用的视频→关节角工具。需要集成第三方视觉工具（见下文路径 B）。

---

## 4. SDG 技术路径总览

```
┌──────────────┬───────────────┬──────────────┬──────────────────┐
│ 路径          │ state/action   │ 图像          │ 难度/成本         │
├──────────────┼───────────────┼──────────────┼──────────────────┤
│ A. 物理仿真   │ 仿真器直接提供  │ 仿真渲染       │ 中（需建仿真场景）  │
│ B. 视频重建   │ IK 推算（近似） │ 原始视频帧     │ 高（精度有限）     │
│ C. World Model│ 已有数据扩展   │ 生成新视觉      │ 高（需训练模型）   │
│ D. 数据增强   │ 复用已有       │ 图像增强       │ 低（立即可用）     │
└──────────────┴───────────────┴──────────────┴──────────────────┘
```

---

## 5. 路径 A：物理仿真采集（最完整）

### 5.1 lerobot 已支持的仿真环境

| 环境 | 安装 | 物理引擎 | 机器人 | 适合 SDG |
|------|------|---------|--------|---------|
| **gym-pusht** | `pip install -e ".[pusht]"` | pymunk (2D) | 推圆盘 | 简单测试 |
| **gym-aloha** | `pip install -e ".[aloha]"` | MuJoCo | 双臂 ALOHA | ✓ |
| **LIBERO** | `pip install -e ".[libero]"` | MuJoCo | Franka 等 | ✓ |
| **MetaWorld** | `pip install -e ".[metaworld]"` | MuJoCo | Sawyer | ✓ |
| **Isaac Lab** | Hub `nvidia/isaaclab-arena-envs` | GPU 仿真 | 通用 | ✓ 最接近真实 |

### 5.2 从仿真环境采集数据写入 LeRobot 格式

lerobot 没有"仿真版 lerobot_record.py"，需要自己写采集循环：

```python
import gymnasium as gym
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 1. 初始化 gym 环境（以 LIBERO 为例）
env = gym.make("libero/LIBERO_SPATIAL_pick_up_the_alphabet_soup_and_place_it_in_the_basket-v0")

# 2. 创建 LeRobotDataset
dataset = LeRobotDataset.create(
    repo_id="your_name/so101_sim_pickplace",
    fps=30,
    robot_type="sim_so101",
    features={
        "observation.state":    {"dtype": "float32", "shape": (6,), "names": ["j0","j1","j2","j3","j4","j5"]},
        "action":               {"dtype": "float32", "shape": (6,), "names": ["j0","j1","j2","j3","j4","j5"]},
        "observation.images.front": {"dtype": "video", "shape": (480, 640, 3), "names": None},
        "task":                 {"dtype": "str",     "shape": (1,), "names": None},
    },
)

# 3. 采集循环
for episode_idx in range(100):
    obs, info = env.reset()
    dataset.start_episode()          # 标记 episode 开始
    done = False
    t = 0

    while not done:
        action = policy.select_action(obs)   # 或脚本化策略
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        frame = {
            "observation.state":    obs["agent_pos"],       # shape (6,)
            "action":               action,                  # shape (6,)
            "observation.images.front": obs["pixels"],       # shape (H,W,3), uint8 → 自动转 float
            "task":                 "pick and place",
            "timestamp":            t / 30.0,
        }
        dataset.add_frame(frame)
        obs = next_obs
        t += 1

    dataset.save_episode(task="pick and place")

dataset.consolidate()
```

### 5.3 SO-101 仿真模型构建路线

1. **获取 URDF**：从 [so-arm100](https://github.com/TheRobotStudio/SO-ARM100) 获取 `so101.urdf`
2. **转为 MuJoCo XML**：使用 `dm_control` 的 `urdf2mjcf` 或手动转换
3. **接入 gymnasium**：基于 `gymnasium-robotics` 或 `dm_control` 包装
4. **渲染视觉**：MuJoCo 支持 RGB 渲染，配置相机位置匹配真实摄像头视角

---

## 6. 路径 B：视频运动重建（Video → State）

### 6.1 完整管线

```python
# 依赖：pip install mediapipe opencv-python placo
# 注：需要 SO-101 URDF 和相机标定参数

import cv2
import numpy as np
from lerobot.model.kinematics import RobotKinematics

kin = RobotKinematics("path/to/so101.urdf", "gripper_frame_link")

# Step 1：视频逐帧提取
cap = cv2.VideoCapture("robot_demo.mp4")

states = []
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Step 2：机器人关节端点 2D 检测
    # 方法 A：MediaPipe Holistic（粗粒度手臂姿态）
    # 方法 B：自定义关键点检测器（精度更高，需要训练数据）
    keypoints_2d = detect_robot_keypoints(frame)   # shape (N, 2)

    # Step 3：2D → 3D（需要相机内参 + 多视角或深度）
    keypoints_3d = triangulate_or_depth(keypoints_2d, camera_matrix, depth_map)

    # Step 4：从 3D 末端位姿求关节角（IK）
    ee_pose = compute_ee_pose_from_keypoints(keypoints_3d)   # 4×4
    joint_angles = kin.inverse_kinematics(ee_pose)           # degrees

    states.append(joint_angles)

# Step 5：相邻帧状态差作为 action（或用 FK 正向预测目标）
actions = states[1:] + [states[-1]]   # 简化：action[t] ≈ state[t+1]
```

### 6.2 可用的第三方工具

| 工具 | 适用场景 | 精度 | 说明 |
|------|---------|------|------|
| **MediaPipe Holistic** | 手/手臂关键点 | 低 | 针对人体设计，机器人外观差异大 |
| **SAM + 深度估计** | 机器人分割 + 3D 重建 | 中 | 需要深度图或立体视觉 |
| **RobotAware / RoboAgent** | 机器人姿态估计 | 中高 | 学术方法，需特定训练数据 |
| **FoundationPose** | 6D 物体姿态 | 高 | 适合已知 3D 模型的末端执行器 |
| **Wrist camera + ArUco** | 末端位姿 | 高 | 简单标定，但需要标记物 |

### 6.3 局限性

- **精度**：关节角误差通常 5-15°，对精密操作任务影响显著
- **夹爪状态**：夹爪开合难以从外部视觉准确恢复
- **适用场景**：适合扩充动作多样性（姿态变化），不适合精密轨迹复现

---

## 7. 路径 C：World Model 生成新观测

### 7.1 思路

给定已有的真实 `(state_t, action_t, image_t)` 三元组，训练一个 World Model 来：
1. **生成新视觉**：固定 state/action 序列，生成不同场景/背景/物体颜色下的图像
2. **生成新轨迹**：在隐空间中插值或扰动，生成新的 state/action 序列 + 对应图像

### 7.2 在 lerobot 体系内的对应工具

| 工具 | 作用 | 来源 |
|------|------|------|
| **TD-MPC2** | 隐式 world model（`src/lerobot/policies/tdmpc/`） | lerobot 内置 |
| **SmolVLA / Pi0** | 将图像+语言→动作，可用于策略采样 | lerobot 内置 |
| **GR00T Blueprint** | NVIDIA 基于 Isaac Sim 的合成数据框架 | 外部，通过 Hub 集成 |

### 7.3 图像级增强 World Model

最实用的 World Model SDG：**保持 state/action 不变，生成不同视觉观测**。

```python
# 方法 1：传统图像增强（Domain Randomization）
from torchvision import transforms
augment = transforms.Compose([
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomPerspective(distortion_scale=0.1),
    transforms.GaussianBlur(kernel_size=3),
])

# 方法 2：背景替换（foreground 分割 + 新背景合成）
# 需要 SAM 分割机器人/物体 foreground，
# 然后与随机背景图/渲染背景合成

# 方法 3：Stable Diffusion inpainting / img2img
# 保留机器人姿态，更换场景风格（需要 controlnet 约束姿态）
```

### 7.4 lerobot 内置的图像变换工具

```python
# src/lerobot/data_processing/image_writer.py
# examples/dataset/use_dataset_image_transforms.py

from lerobot.datasets.image_transforms import get_image_transforms

transforms = get_image_transforms(
    color_jitter={"brightness": 0.3, "contrast": 0.3},
    random_crop={"crop_params": (0.8, 1.0)},
    normalize=True,
)
```

---

## 8. 路径 D：数据增强（最快上手）

### 8.1 LeRobot 内置图像增强

已直接集成到 `LeRobotDataset` 的 transform 机制：

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torchvision import transforms

dataset = LeRobotDataset(
    "lerobot/svla_so101_pickplace",
    image_transforms=transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.0),   # 机器人不适合水平翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomErasing(p=0.1),           # 模拟遮挡
    ]),
)
```

### 8.2 轨迹级增强

```python
import torch
import numpy as np

def augment_trajectory(state_seq, action_seq, noise_std=0.5):
    """
    对关节角轨迹加高斯噪声（单位：度）。
    注意：noise_std=0.5度 对于精密操作是合理范围。
    """
    state_aug = state_seq + np.random.normal(0, noise_std, state_seq.shape)
    action_aug = action_seq + np.random.normal(0, noise_std, action_seq.shape)
    return state_aug, action_aug

def time_warp(state_seq, action_seq, speed_factor=0.8):
    """
    对轨迹进行时间弯曲（变速）。speed_factor < 1 表示变慢。
    """
    T = state_seq.shape[0]
    new_T = int(T / speed_factor)
    t_orig = np.linspace(0, 1, T)
    t_new = np.linspace(0, 1, new_T)
    state_aug = np.array([np.interp(t_new, t_orig, state_seq[:, d]) for d in range(6)]).T
    action_aug = np.array([np.interp(t_new, t_orig, action_seq[:, d]) for d in range(6)]).T
    return state_aug, action_aug
```

---

## 9. 写入 LeRobot 格式的 API

无论采用哪种 SDG 路径，最终都需要把数据写入 LeRobot 格式：

```python
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 1. 创建数据集
dataset = LeRobotDataset.create(
    repo_id="your_name/so101_synthetic",
    fps=30,
    robot_type="so101",
    features={
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["shoulder_pan", "shoulder_lift", "elbow_flex",
                      "wrist_flex", "wrist_roll", "gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["shoulder_pan", "shoulder_lift", "elbow_flex",
                      "wrist_flex", "wrist_roll", "gripper"],
        },
        "observation.images.up": {
            "dtype": "video",
            "shape": (480, 640, 3),   # HWC
            "names": ["height", "width", "channel"],
        },
        "observation.images.side": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"],
        },
    },
    image_writer_threads=4,
)

# 2. 逐 episode 写入
for episode_data in synthetic_episodes:
    for t, frame_data in enumerate(episode_data):
        frame = {
            "observation.state":          frame_data["state"],         # np.ndarray (6,)
            "action":                     frame_data["action"],         # np.ndarray (6,)
            "observation.images.up":      frame_data["image_up"],       # np.ndarray (H,W,3) uint8
            "observation.images.side":    frame_data["image_side"],     # np.ndarray (H,W,3) uint8
            "task":                       "pick up object and place",
        }
        dataset.add_frame(frame)

    dataset.save_episode(task="pick up object and place")

# 3. 计算统计并保存
dataset.consolidate(run_compute_stats=True)

# 4. 可选：推送到 HuggingFace Hub
# dataset.push_to_hub()
```

### 9.1 合并真实数据 + 合成数据

```python
from lerobot.datasets.dataset_tools import merge_datasets

# 使用 lerobot-edit-dataset CLI
# lerobot-edit-dataset \
#   --repo_id your_name/so101_merged \
#   --operation.type merge \
#   --operation.repo_ids "['lerobot/svla_so101_pickplace', 'your_name/so101_synthetic']"

# 或 Python API
merge_datasets(
    repo_ids=["lerobot/svla_so101_pickplace", "your_name/so101_synthetic"],
    output_repo_id="your_name/so101_merged",
)
```

---

## 10. 推荐起步方案

### 优先级排序

```
短期（1-2周）：路径 D 数据增强
├── 图像增强（颜色抖动、随机遮挡）+ 轨迹噪声
├── 直接基于 svla_so101_pickplace 扩充
└── 成本极低，能快速验证 policy 对增强数据的响应

中期（1-2月）：路径 A 仿真采集（LIBERO 或 MuJoCo）
├── 目标：搭建 SO-101 MuJoCo 仿真 + 随机化场景
├── 依赖：SO-101 URDF → MuJoCo XML + 脚本化策略
└── 产出：大规模仿真数据（理论无上限）

长期：路径 C World Model
├── 基于 lerobot/smolvla_base 或 Pi0 的隐式 world model
├── 生成新视觉场景下的 rollout 数据
└── 需要验证生成数据的有效性（sim-to-real gap）
```

### 快速验证脚本（路径 D）

```python
# 01_beginner/gen_augmented_dataset.py

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torchvision import transforms
import torch

# 加载原始数据
dataset = LeRobotDataset("lerobot/svla_so101_pickplace", episodes=[0, 1, 2])

# 图像增强变换
aug = transforms.Compose([
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
])

# 检验增强效果
sample = dataset[100]
img = sample["observation.images.up"]           # (C, H, W) float32 in [0,1]
img_aug = aug(img)                               # 增强后图像

# 验证 state/action 维度和值域
state = sample["observation.state"]
action = sample["action"]
print(f"state shape: {state.shape}, range: [{state.min():.1f}, {state.max():.1f}] degrees")
print(f"action shape: {action.shape}, range: [{action.min():.1f}, {action.max():.1f}] degrees")
print(f"state - action (tracking error): {(state - action).abs().mean():.2f} degrees")
```

---

---

## 11. Genesis 仿真引擎可行性评估

### 11.1 Genesis 与 SO-101 的直接关联

GitHub Issue [#1858](https://github.com/Genesis-Embodied-AI/Genesis/issues/1858) 明确展示了 Genesis + SO-101 + lerobot 的集成：

```python
# 直接导入 lerobot 的 SO-101 遥操作器
from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader

# 加载 SO-101 MuJoCo XML 模型
so101 = scene.add_entity(
    gs.morphs.MJCF(file="assets/so101_new_calib.xml"),
)

# 关节名与 lerobot 完全一致
joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# 角度单位转换（lerobot 用度，Genesis 物理引擎用弧度）
def convert_action_to_genesis(action):
    return np.array([
        np.deg2rad(action["shoulder_pan.pos"]),
        np.deg2rad(action["shoulder_lift.pos"]),
        ...
    ])
```

**结论：Genesis 与 LeRobot 的数据格式天然兼容，集成成本极低。**

### 11.2 Genesis 核心优势（对比现有方案）

| 维度 | MuJoCo（robosim-so101） | Isaac Sim（so101-autogen） | **Genesis** |
|------|------------------------|--------------------------|-------------|
| 速度 | ~1x | ~1x (GPU) | **10-80x faster** |
| 并行环境 | 有限 | 有限 | **数千个并行环境** |
| 安装难度 | 简单 | 复杂（需 Isaac Sim） | **pip install genesis-world** |
| Python 原生 | 部分 | 部分 | **100% Python** |
| 物理精度 | 高 | 高 | 高（同等级） |
| 可变形物体 | 无 | 有限 | **MPM（弹性/塑性/流体）** |
| 渲染质量 | 低 | 高 | **光线追踪** |
| SO-101 支持 | ✓（MJCF） | ✓ | **✓（同一套 MJCF）** |

### 11.3 社区已有验证案例

| 项目 | 说明 | 规模 |
|------|------|------|
| [robosim-so101-pickup-v2](https://huggingface.co/datasets/houmans/robosim-so101-pickup-v2) | MuJoCo 仿真，SO-101 pick-up，LeRobot v3 格式 | 400 episodes, 93k frames |
| [robosim-so101-pickup-v3](https://huggingface.co/datasets/houmans/robosim-so101-pickup-v3) | 同上 v3 版本 | 100 episodes |
| [so101-autogen](https://github.com/haoran1062/so101-autogen) | Isaac Sim + IK + 状态机，输出 LeRobot 格式 | 开源框架 |

这些项目证明了"仿真 → LeRobot 格式"的完整管线是可行的，且 SO-101 数据格式已经打通。

### 11.4 Genesis SDG 管线设计

**目标**：用 Genesis 并行仿真生成大规模 SO-101 pick-place 数据集，直接写入 LeRobot 格式。

```
┌─────────────────────────────────────────────────────────────────┐
│                 Genesis + LeRobot SDG 管线                       │
│                                                                 │
│  1. 场景构建                                                      │
│     Genesis Scene + SO-101 MJCF + 随机化物体（位置/大小/颜色）     │
│     N 个并行环境（batch_size=256）                                  │
│                                                                 │
│  2. 轨迹生成                                                      │
│     IK 规划 → 位置控制 → so101.control_dofs_position()            │
│     状态机：Home → Pre-grasp → Approach → Grasp → Lift           │
│                                                                 │
│  3. 数据记录                                                      │
│     每帧读取 joint_pos（degrees） → observation.state              │
│     每帧记录目标 joint_pos → action                                │
│     Camera.render() → observation.images.up / side               │
│                                                                 │
│  4. 写入 LeRobot 格式                                             │
│     dataset.add_frame() → save_episode() → consolidate()         │
│     可选 push_to_hub()                                            │
└─────────────────────────────────────────────────────────────────┘
```

### 11.5 关键实现要点

#### 坐标系与单位

```python
# Genesis 物理引擎内部用弧度，lerobot 数据集用度
# 读取关节状态时需要转换
import genesis as gs
import numpy as np

dof_positions_rad = so101.get_dofs_position(dof_indices)     # 弧度（Genesis 返回）
state_degrees = np.rad2deg(dof_positions_rad.cpu().numpy())  # → 度（LeRobot 格式）

# 写入 frame 时
frame = {
    "observation.state": state_degrees,   # 度
    "action":            target_degrees,  # 度（IK 规划的目标）
    ...
}
```

#### 并行采集框架（伪代码）

```python
import genesis as gs
from lerobot.datasets.lerobot_dataset import LeRobotDataset

gs.init(backend=gs.gpu)

N_ENVS = 256   # 256 个并行环境，Genesis 轻松处理

scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1/500, substeps=10))
so101 = scene.add_entity(gs.morphs.MJCF("so101_new_calib.xml"))
cam_up   = scene.add_camera(res=(640, 480), pos=(...), lookat=(...))
cam_side = scene.add_camera(res=(640, 480), pos=(...), lookat=(...))

# 随机化物体
cubes = []
for i in range(N_ENVS):
    cube = scene.add_entity(
        gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=randomize_pos()),
        surface=gs.surfaces.Default(color=random_color()),
    )
    cubes.append(cube)

scene.build(n_envs=N_ENVS)

dataset = LeRobotDataset.create(...)

for ep in range(N_ENVS):
    # 每个并行 env 对应一个 episode
    trajectory = plan_ik_trajectory(cubes[ep].get_pos())

    for t, target_deg in enumerate(trajectory):
        target_rad = np.deg2rad(target_deg)
        so101.control_dofs_position(target_rad[ep], dof_indices)
        scene.step()

        # 记录状态
        state_deg = np.rad2deg(so101.get_dofs_position(dof_indices)[ep].cpu().numpy())
        img_up   = cam_up.render()[ep]    # (H, W, 3) uint8
        img_side = cam_side.render()[ep]

        dataset.add_frame({
            "observation.state":         state_deg,
            "action":                    target_deg,
            "observation.images.up":     img_up,
            "observation.images.side":   img_side,
            "task": "pick up the cube and place it in the box",
        })
    dataset.save_episode(...)
```

### 11.6 可行性结论

| 评估维度 | 结论 | 风险 |
|---------|------|------|
| **技术可行性** | ✅ 完全可行，社区已有 MuJoCo 版本验证 | 低 |
| **SO-101 MJCF 模型** | ✅ 已有（同 MuJoCo 格式），Genesis 直接加载 | 低 |
| **单位/格式兼容** | ✅ 角度制，关节名完全对齐 | 低（注意 deg↔rad 转换） |
| **LeRobot 格式输出** | ✅ 直接用 `add_frame()` + `save_episode()` | 低 |
| **规模化生产** | ✅ 256+ 并行环境，10-80x 加速 | 低 |
| **视觉质量** | ⚠️ 光线追踪渲染出色，但 sim-to-real gap 仍存在 | 中 |
| **IK 规划器** | ⚠️ 需要自己实现或复用 `placo`（lerobot 已集成） | 中 |
| **弹性物体抓取** | ⚠️ Issue #1858 报告 MPM 物体无法抓取，需等 Genesis 修复 | 中（刚性物体已 OK）|

**总结**：Genesis 是目前最适合 SO-101 SDG 的仿真引擎：
- 安装最简单（`pip install genesis-world`）
- 与 lerobot 接口天然兼容
- 并行仿真速度是最大优势（可快速生成百万级数据）
- 刚性物体（积木、方块）抓取任务已完全可行
- 弹性/软体物体抓取需关注 #1858 的修复进展

### 11.7 推荐落地步骤

```
Week 1: 环境搭建 + 基础验证
├── pip install genesis-world
├── 加载 so101_new_calib.xml，验证关节控制
├── 实现 lerobot 格式的单 episode 记录
└── 对比 Genesis 状态数据 与 svla_so101_pickplace 数据分布

Week 2-3: IK 轨迹规划 + 状态机
├── 集成 placo IK（lerobot 已有）或 Genesis 自带 IK
├── 实现 pick-place 状态机（Home→Pre-grasp→Approach→Grasp→Lift→Place）
└── 单环境成功率验证

Week 4: 并行扩展 + 域随机化
├── 扩展到 N_ENVS=256 并行环境
├── 物体位置/大小/颜色/质量随机化
└── 批量生成 1000+ episodes，推送到 HuggingFace

后续: 用合成数据微调 SmolVLA，验证 sim-to-real 迁移效果
```

---

## 参考资料

- [LeRobot Dataset Format v3.0](https://github.com/huggingface/lerobot/blob/main/docs/datasets.md)
- [SO-ARM100 URDF](https://github.com/TheRobotStudio/SO-ARM100)
- [placo 运动学库](https://github.com/Rhoban/placo)
- [Genesis 文档](https://genesis-world.readthedocs.io/)
- [Genesis Issue #1858：SO-101 in Genesis](https://github.com/Genesis-Embodied-AI/Genesis/issues/1858)
- [robosim-so101-pickup-v2（MuJoCo 版 SO-101 数据集）](https://huggingface.co/datasets/houmans/robosim-so101-pickup-v2)
- [so101-autogen（Isaac Sim 版自动化数据生成）](https://github.com/haoran1062/so101-autogen)
- [LIBERO 仿真基准](https://libero-project.github.io/)
- [GR00T Isaac Sim SDG](https://developer.nvidia.com/isaac/gr00t)
- [FoundationPose 6D 姿态估计](https://github.com/NVlabs/FoundationPose)
- [gym-aloha](https://github.com/huggingface/gym-aloha) / [gym-pusht](https://github.com/huggingface/gym-pusht)
