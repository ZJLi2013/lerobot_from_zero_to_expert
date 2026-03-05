# Genesis x SO-101 SDG Implementation Guide

> 面向“可跑通、可复现、可调参”的实现文档。

---

## 1. 目标与范围

本指南聚焦 3 件事：

1. 在 Genesis 中稳定运行 SO-101 抓取数据采集
2. 输出可分析数据（`npy + .rrd + metrics.json`）
3. 输出可训练数据（LeRobot v3：`parquet + mp4 + meta`）

不再展开长篇背景对比，细节决策请参考：

- `best_practices.md`（调参与验收）
- `scripts/reports.md`（实验记录）

---

## 2. 项目结构（与本主题相关）

```text
02_intermediate/
├── readme.md
├── genesis_sdg.md
├── best_practices.md
├── sdg_data/                      # 输出数据（已在 .gitignore）
└── scripts/
    ├── poc_genesis_pipeline.py
    ├── sdg_so101_genesis.py
    ├── sdg_so101_improved.py
    ├── sdg_so101_grasp_experiment.py
    ├── viz_sdg_rerun.py
    ├── npy_to_lerobot.py
    └── reports.md
```

---

## 3. 环境要求

## 3.1 本地

- Windows + Git + SSH
- 可选：Rerun（查看 `.rrd`）

## 3.2 4090 远端

- Docker + NVIDIA runtime
- 镜像：`genesis_poc:latest`
- 仓库路径：`~/github/lerobot_from_zero_to_expert`

---

## 4. 脚本选择建议

## 4.1 快速检查链路

使用：`scripts/poc_genesis_pipeline.py`

适用场景：先确认“Genesis 能跑 + 能出数据”，不追求抓取质量。

## 4.2 基础采集

使用：`scripts/sdg_so101_genesis.py`

适用场景：验证 SO-101 URDF + 双相机 + npy 输出结构。

## 4.3 改进采集

使用：`scripts/sdg_so101_improved.py`

适用场景：加入 probe、朝向/姿态改进、内置 `.rrd` 输出。

## 4.4 抓取调参实验（推荐）

使用：`scripts/sdg_so101_grasp_experiment.py`

适用场景：1 episode 快速迭代抓取参数；支持自动 offset 搜索与 `metrics.json` 记录。

---

## 5. 最小运行流程（本地 -> 4090 -> 回传）

## 5.1 推送本地修改

```powershell
cd "c:\Users\zhengjli\Documents\github\lerobot_from_zero_to_expert"
git add .
git commit -m "update sdg docs/scripts"
git push origin main
```

## 5.2 4090 拉取并运行（1 episode，自动 offset）

```bash
ssh david@<4090_HOST> "cd ~/github/lerobot_from_zero_to_expert && git pull origin main"

ssh david@<4090_HOST> "mkdir -p ~/sdg_grasp_exp && docker run --rm --gpus all \
  -v ~/github/lerobot_from_zero_to_expert:/workspace/lfzte \
  -v ~/sdg_grasp_exp:/output \
  genesis_poc:latest \
  python -u /workspace/lfzte/02_intermediate/scripts/sdg_so101_grasp_experiment.py \
  --exp-id E3_auto_offset \
  --episodes 1 \
  --episode-length 6 \
  --save /output \
  --auto-tune-offset \
  --offset-x-candidates=-0.008,-0.004,0.0,0.004,0.008 \
  --offset-y-candidates=-0.010,-0.005,0.0,0.005,0.010 \
  --gripper-open 70 \
  --gripper-close 20 \
  --close-hold-steps 12"
```

## 5.3 回传结果

```powershell
scp david@<4090_HOST>:~/sdg_grasp_exp/E3_auto_offset/improved_sdg_E3_auto_offset.rrd `
  "c:\Users\zhengjli\Documents\github\lerobot_from_zero_to_expert\02_intermediate\sdg_data\improved_sdg_E3_auto_offset.rrd"

scp david@<4090_HOST>:~/sdg_grasp_exp/E3_auto_offset/metrics.json `
  "c:\Users\zhengjli\Documents\github\lerobot_from_zero_to_expert\02_intermediate\sdg_data\metrics_E3_auto_offset.json"
```

---

## 6. 数据输出说明

## 6.1 调参实验输出

- `improved_sdg_<exp_id>.rrd`：可视化回放
- `metrics.json`：抓取判定与参数记录
- `states/actions/images_*.npy`：原始数值与图像

## 6.2 LeRobot v3 输出

通过 `scripts/npy_to_lerobot.py` 转换后得到：

- `data/chunk-xxx/file-xxx.parquet`
- `videos/.../episode_xxxxxx.mp4`
- `meta/info.json`, `meta/tasks.parquet`, `meta/episodes/*.parquet`

---

## 7. 统一验收标准

以 `metrics.json` 为主：

- `grasp_success == 1`
- `cube_lift_delta > 0.01m`

辅助检查：

- `state/action` 在关节限位范围内
- `.rrd` 中 box 在 lift 阶段跟随夹爪上升

---

## 8. 常见问题（短版）

- **Q: 朝向对了但抓不到？**  
  A: 先调 `offset_y`，再调 `offset_x`，最后调 `close/hold`。

- **Q: 自动 offset 仍失败？**  
  A: 优先确认 `ee_link` 是否真的使用 `gripperframe`；再扩大 offset 搜索范围并联合搜索 `close/hold`。

- **Q: 只想快速验证链路？**  
  A: 先跑 `poc_genesis_pipeline.py`，确认环境无误后再跑抓取实验。

---

## 9. 文档分工

- 入口导航：`readme.md`
- 实现流程：`genesis_sdg.md`（本文件）
- 调参与判定：`best_practices.md`
- 历史记录：`scripts/reports.md`

# Genesis × SO-101 合成数据生成（SDG）完整指南

> **目标**：使用 Genesis 仿真引擎为 SO-101 机械臂批量生成 pick-place 合成数据集，格式直接兼容 LeRobot v2/v3，用于训练 SmolVLA / ACT 等端到端策略。

---

## 目录

1. [可行性评估](#1-可行性评估)
   - 1.1 Genesis 与 SO-101 的直接关联
   - 1.2 核心优势 vs 对比方案
   - 1.3 社区已有验证案例
   - 1.4 技术栈兼容性
   - 1.5 风险清单
   - 1.6 可行性结论
2. [架构设计](#2-架构设计)
3. [环境准备](#3-环境准备)
4. [核心 Sample Code](#4-核心-sample-code)
   - 4.1 单环境验证脚本
   - 4.2 并行批量采集脚本（完整版）
5. [关键设计决策](#5-关键设计决策)
6. [已知问题与规避](#6-已知问题与规避)
7. [落地步骤](#7-落地步骤)

---

## 1. 可行性评估

### 1.1 Genesis 与 SO-101 的直接关联

GitHub Issue [#1858](https://github.com/Genesis-Embodied-AI/Genesis/issues/1858) 明确展示了 Genesis + SO-101 + lerobot 的完整集成：

```python
# 直接导入 lerobot 的 SO-101 遥操作器
from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader

# 加载 SO-101 MuJoCo XML 模型（与 MuJoCo 版本完全相同的文件）
so101 = scene.add_entity(
    gs.morphs.MJCF(file="assets/so101_new_calib.xml"),
)

# 关节名与 lerobot 数据集完全一致
joints = ["shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper"]

# 角度单位转换（lerobot 用度，Genesis 物理引擎用弧度）
def convert_action_to_genesis(action):
    return np.array([
        np.deg2rad(action["shoulder_pan.pos"]),
        np.deg2rad(action["shoulder_lift.pos"]),
        np.deg2rad(action["elbow_flex.pos"]),
        np.deg2rad(action["wrist_flex.pos"]),
        np.deg2rad(action["wrist_roll.pos"]),
        np.deg2rad(action["gripper.pos"]),
    ])
```

**结论：Genesis 与 LeRobot 的数据格式天然兼容，集成成本极低。**

### 1.2 核心优势 vs 对比方案

| 维度 | MuJoCo（robosim-so101） | Isaac Sim（so101-autogen） | **Genesis** |
|------|------------------------|--------------------------|-------------|
| 安装难度 | 简单 | 复杂（需 Isaac Sim） | **`pip install genesis-world`** |
| 速度 | ~1x | ~1x (GPU) | **10-80x faster** |
| 并行环境 | 有限 | 有限 | **数千个并行环境** |
| Python 原生 | 部分 | 部分 | **100% Python** |
| 物理精度 | 高 | 高 | 高（同等级） |
| 可变形物体 | 无 | 有限 | **MPM（弹性/塑性/流体）** |
| 渲染质量 | 低 | 高 | **光线追踪** |
| SO-101 支持 | ✓（MJCF） | ✓ | **✓（同一套 MJCF）** |

### 1.3 社区已有验证案例

| 项目 | 说明 | 规模 |
|------|------|------|
| [Genesis Issue #1858](https://github.com/Genesis-Embodied-AI/Genesis/issues/1858) | SO-101 + lerobot + Genesis 集成完整演示 | — |
| [robosim-so101-pickup-v2](https://huggingface.co/datasets/houmans/robosim-so101-pickup-v2) | MuJoCo 仿真，SO-101 pick-up，LeRobot v3 格式 | 400 episodes, 93k frames |
| [robosim-so101-pickup-v3](https://huggingface.co/datasets/houmans/robosim-so101-pickup-v3) | 同上 v3 版本 | 100 episodes |
| [so101-autogen](https://github.com/haoran1062/so101-autogen) | Isaac Sim + IK + 状态机，输出 LeRobot 格式 | 开源框架 |

这些项目证明了"仿真 → LeRobot 格式"的完整管线是可行的，且 SO-101 数据格式已经打通。Genesis 是其中安装最简单、并行速度最快的选项。

### 1.4 技术栈兼容性

| 层次 | 工具 | 状态 |
|------|------|------|
| 仿真引擎 | Genesis (`pip install genesis-world`) | ✅ |
| 机器人模型 | SO-101 MuJoCo XML（`so101_new_calib.xml`） | ✅ 直接加载 |
| 数据格式 | LeRobot `LeRobotDataset` v2/v3 | ✅ |
| 单位约定 | Genesis 弧度 ↔ LeRobot 角度制 | ✅ 手动转换 |
| 并行仿真 | Genesis `scene.build(n_envs=N)` | ✅ |
| 图像渲染 | Rasterizer / BatchRenderer / RayTracer | ✅ |
| 逆运动学 | Genesis `robot.inverse_kinematics()` | ✅ |

### 1.5 风险清单

| 风险 | 等级 | 规避方案 |
|------|------|---------|
| deg ↔ rad 转换遗漏 | 低 | 封装统一工具函数 |
| IK 收敛失败 | 中 | 多次重试 + 种子随机化 |
| MPM 软体物体无法被夹爪抓取（Issue #1858） | 中 | **刚性方块优先**，等待修复 |
| Sim-to-Real gap | 中 | 域随机化（位置/颜色/光照/质量） |
| 多环境相机渲染显存 | 中 | 分批渲染，BatchRenderer |

### 1.6 可行性结论

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

---

## 2. 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│              Genesis SO-101 SDG 管线                              │
│                                                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐   │
│  │  场景构建    │    │  轨迹规划    │    │   数据写入           │   │
│  │             │    │             │    │                     │   │
│  │ • SO-101    │───▶│ • IK 求解   │───▶│ • observation.state │   │
│  │   MJCF 加载 │    │ • 状态机    │    │ • action (degrees)  │   │
│  │ • 目标物体  │    │   Home      │    │ • images.top        │   │
│  │   域随机化  │    │   PreGrasp  │    │ • images.wrist      │   │
│  │ • N 个并行  │    │   Approach  │    │ • task (文本)        │   │
│  │   环境      │    │   Grasp     │    │                     │   │
│  │             │    │   Lift      │    │ LeRobotDataset      │   │
│  │             │    │   Place     │    │ .add_frame()        │   │
│  └─────────────┘    └─────────────┘    │ .save_episode()     │   │
│                                        └─────────────────────┘   │
│                                                                   │
│  输出：HuggingFace Dataset（可直接用 lerobot train 命令训练）       │
└─────────────────────────────────────────────────────────────────┘
```

### 关节定义（与 LeRobot SO-101 完全对齐）

```python
# SO-101 有 6 个自由度（lerobot 惯例，角度制）
JOINT_NAMES = [
    "shoulder_pan",   # 0
    "shoulder_lift",  # 1
    "elbow_flex",     # 2
    "wrist_flex",     # 3
    "wrist_roll",     # 4
    "gripper",        # 5 (0=open, max=closed)
]

# LeRobot 数据集 motor names
MOTOR_NAMES = [f"{j}.pos" for j in JOINT_NAMES]
# → ["shoulder_pan.pos", "shoulder_lift.pos", ...]
```

---

## 3. 环境准备

```bash
# 1. 安装 Genesis
pip install genesis-world

# 2. 安装 LeRobot（含数据集 API）
pip install lerobot

# 3. 下载 SO-101 MuJoCo XML（从 lerobot 或 SO-ARM100 仓库）
# lerobot 已内置：
python -c "import lerobot; print(lerobot.__file__)"
# 找到 lerobot/common/robot_devices/robots/assets/so101_new_calib.xml

# 或手动克隆：
# git clone https://github.com/TheRobotStudio/SO-ARM100
# cp SO-ARM100/URDF/so101/so101_new_calib.xml ./assets/

# 4. 验证 Genesis 安装
python -c "import genesis as gs; gs.init(); print('Genesis OK')"
```

---

## 4. 核心 Sample Code

### 4.1 单环境验证脚本

用于在批量采集前验证 SO-101 加载、IK、关节控制、相机渲染均正常。

```python
# validate_so101_genesis.py
"""
单环境验证：确认 SO-101 在 Genesis 中可正常控制并记录数据。
运行：python validate_so101_genesis.py --vis
"""
import argparse
import numpy as np
import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vis", action="store_true", help="显示实时视图")
    parser.add_argument(
        "--xml",
        default="assets/so101_new_calib.xml",
        help="SO-101 MJCF 路径",
    )
    args = parser.parse_args()

    # ── 初始化 ──────────────────────────────────────────────────────────
    gs.init(backend=gs.gpu, logging_level="warning")

    # ── 场景 ────────────────────────────────────────────────────────────
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1 / 50, substeps=10),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            enable_joint_limit=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, -0.5, 0.4),
            camera_lookat=(0.0, 0.0, 0.15),
            camera_fov=45,
        ),
        show_viewer=args.vis,
    )

    # ── 实体 ────────────────────────────────────────────────────────────
    scene.add_entity(gs.morphs.Plane())

    so101 = scene.add_entity(
        gs.morphs.MJCF(
            file=args.xml,
            pos=(0.0, 0.0, 0.0),
        )
    )

    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.03, 0.03, 0.03),
            pos=(0.15, 0.0, 0.015),
        ),
        surface=gs.surfaces.Default(color=(1.0, 0.3, 0.3, 1.0)),
    )

    # ── 相机 ────────────────────────────────────────────────────────────
    cam_top = scene.add_camera(
        res=(640, 480),
        pos=(0.0, 0.0, 0.6),
        lookat=(0.15, 0.0, 0.0),
        fov=60,
        GUI=args.vis,
    )

    # ── 构建（单环境） ───────────────────────────────────────────────────
    scene.build()

    # ── 关节索引：SO-101 有 6 DOF（顺序与 JOINT_NAMES 一致）─────────────
    n_arm_dofs = 5   # shoulder_pan ~ wrist_roll
    n_dofs_total = 6
    arm_dof_idx = np.arange(n_arm_dofs)
    gripper_dof_idx = np.array([5])
    all_dof_idx = np.arange(n_dofs_total)

    # PD 增益（SO-101 参考值，可根据实际调整）
    so101.set_dofs_kp(np.array([100, 100, 80, 80, 60, 40]), all_dof_idx)
    so101.set_dofs_kv(np.array([10, 10, 8, 8, 6, 4]),   all_dof_idx)

    # ── home 姿态（弧度） ────────────────────────────────────────────────
    HOME_RAD = np.deg2rad([0.0, -30.0, 90.0, -60.0, 0.0, 0.0])

    so101.set_qpos(HOME_RAD)
    for _ in range(50):
        scene.step()

    # ── 末端执行器链接 ───────────────────────────────────────────────────
    # 根据实际 XML 调整链接名称
    ee_link = so101.get_link("gripper_link")

    # ── IK 求解：移动到方块上方 ──────────────────────────────────────────
    cube_pos = cube.get_pos().cpu().numpy()[0]   # (3,)
    pregrasp_pos = cube_pos + np.array([0.0, 0.0, 0.08])

    qpos_pre = so101.inverse_kinematics(
        link=ee_link,
        pos=pregrasp_pos,
        quat=np.array([0.0, 1.0, 0.0, 0.0]),   # 末端朝下
    )
    qpos_pre[gripper_dof_idx] = 0.0  # 夹爪张开

    # ── 执行轨迹并记录关键帧 ────────────────────────────────────────────
    frames = []

    def step_and_record(target_rad: np.ndarray, n_steps: int = 30):
        so101.control_dofs_position(target_rad, all_dof_idx)
        for _ in range(n_steps):
            scene.step()
            cur_rad = so101.get_dofs_position(all_dof_idx).cpu().numpy()
            cur_deg = np.rad2deg(cur_rad)
            tgt_deg = np.rad2deg(target_rad)
            rgb, _, _, _ = cam_top.render(rgb=True, depth=False,
                                          segmentation=False, normal=False)
            img = rgb.cpu().numpy().astype(np.uint8)   # (H, W, 3)
            frames.append({
                "observation.state": cur_deg.tolist(),
                "action":            tgt_deg.tolist(),
                "image":             img,
            })

    # 执行 Pre-Grasp → Approach → Grasp → Lift
    step_and_record(qpos_pre, n_steps=50)

    qpos_grasp = so101.inverse_kinematics(
        link=ee_link,
        pos=cube_pos + np.array([0.0, 0.0, 0.02]),
        quat=np.array([0.0, 1.0, 0.0, 0.0]),
    )
    qpos_grasp[gripper_dof_idx] = 0.0
    step_and_record(qpos_grasp, n_steps=30)

    # 闭合夹爪
    qpos_close = qpos_grasp.copy()
    qpos_close[gripper_dof_idx] = np.deg2rad(25.0)   # 根据 XML 调整最大值
    step_and_record(qpos_close, n_steps=20)

    qpos_lift = so101.inverse_kinematics(
        link=ee_link,
        pos=cube_pos + np.array([0.0, 0.0, 0.15]),
        quat=np.array([0.0, 1.0, 0.0, 0.0]),
    )
    qpos_lift[gripper_dof_idx] = qpos_close[gripper_dof_idx]
    step_and_record(qpos_lift, n_steps=40)

    # ── 打印验证信息 ─────────────────────────────────────────────────────
    print(f"✓ 共记录 {len(frames)} 帧")
    s = np.array(frames[0]["observation.state"])
    a = np.array(frames[0]["action"])
    print(f"  state shape={s.shape}, range=[{s.min():.1f}, {s.max():.1f}] deg")
    print(f"  action shape={a.shape}, range=[{a.min():.1f}, {a.max():.1f}] deg")
    print(f"  image shape={frames[0]['image'].shape}")
    print("✓ 验证通过，可以进行批量采集")


if __name__ == "__main__":
    main()
```

---

### 4.2 并行批量采集脚本（完整版）

这是核心生产脚本，支持 N 个并行环境、域随机化、LeRobot 格式写入。

```python
# genesis_so101_sdg.py
"""
Genesis × SO-101 合成数据批量生成脚本

功能：
- N_ENVS 个并行仿真环境
- pick-place 状态机（Home→PreGrasp→Approach→Grasp→Lift→Place→Return）
- 域随机化（目标物体位置/颜色/质量）
- 双相机（俯视 + 腕部）图像采集
- 输出 LeRobot HuggingFace Dataset 格式

用法：
    python genesis_so101_sdg.py \
        --n_envs 64 \
        --n_episodes 500 \
        --repo_id your_hf_username/so101-genesis-pickplace \
        --push_to_hub
"""

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import genesis as gs
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# ── 常量 ─────────────────────────────────────────────────────────────────────

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# LeRobot motor 名（角度制，单位：度）
MOTOR_NAMES = [f"{j}.pos" for j in JOINT_NAMES]

# SO-101 HOME 姿态（度）
HOME_DEG = np.array([0.0, -30.0, 90.0, -60.0, 0.0, 0.0])

# 夹爪：0=张开，>0=闭合（根据实际 XML 调整）
GRIPPER_OPEN_DEG   = 0.0
GRIPPER_CLOSED_DEG = 25.0

# 关节索引
ARM_DOF_IDX     = np.arange(5)    # shoulder_pan … wrist_roll
GRIPPER_DOF_IDX = np.array([5])
ALL_DOF_IDX     = np.arange(6)


# ── 配置 ─────────────────────────────────────────────────────────────────────

@dataclass
class SDGConfig:
    # 仿真
    xml_path:     str   = "assets/so101_new_calib.xml"
    n_envs:       int   = 64
    ctrl_dt:      float = 1 / 50       # 50 Hz 控制频率
    substeps:     int   = 10

    # 采集
    n_episodes:   int   = 500
    max_steps_per_episode: int = 400   # 每 episode 最多步数

    # 域随机化
    cube_x_range: tuple = (0.10, 0.25)   # 前后范围 (m)
    cube_y_range: tuple = (-0.10, 0.10)  # 左右范围 (m)
    cube_size_range: tuple = (0.025, 0.040)  # 边长范围 (m)

    # 数据集
    repo_id:      str   = "local/so101-genesis-pickplace"
    fps:          int   = 50
    push_to_hub:  bool  = False

    # 渲染
    img_width:    int   = 640
    img_height:   int   = 480
    use_wrist_cam: bool = True    # 是否渲染腕部相机
    show_viewer:  bool  = False


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def deg2rad_batch(deg: np.ndarray) -> np.ndarray:
    """度 → 弧度，支持 (6,) 或 (N, 6)"""
    return np.deg2rad(deg)

def rad2deg_batch(rad) -> np.ndarray:
    """弧度（torch.Tensor 或 np.ndarray） → 度"""
    if isinstance(rad, torch.Tensor):
        rad = rad.cpu().numpy()
    return np.rad2deg(rad)

def random_color() -> tuple:
    r, g, b = np.random.uniform(0.2, 1.0, 3)
    return (float(r), float(g), float(b), 1.0)

def interpolate_joints(
    start: np.ndarray,
    end:   np.ndarray,
    n_steps: int,
) -> list[np.ndarray]:
    """在 start 和 end 之间线性插值 n_steps 个路径点（含终点）"""
    return [
        start + (end - start) * (i + 1) / n_steps
        for i in range(n_steps)
    ]


# ── 状态机 ───────────────────────────────────────────────────────────────────

class PickPlaceStateMachine:
    """
    简单 pick-place 状态机，为单个 env 生成关节角度序列（度）。
    返回 list of (target_deg, gripper_state)
    """

    def __init__(self, so101, ee_link, cube_pos_3d: np.ndarray):
        self.so101     = so101
        self.ee_link   = ee_link
        self.cube_pos  = cube_pos_3d  # (3,) world frame

    def solve_ik_deg(
        self,
        pos:         np.ndarray,
        quat:        np.ndarray,
        gripper_deg: float,
        env_idx:     int = 0,
    ) -> Optional[np.ndarray]:
        """IK 求解，返回 6-DOF 目标角度（度），失败返回 None"""
        try:
            q = self.so101.inverse_kinematics(
                link=self.ee_link,
                pos=pos,
                quat=quat,
                envs_idx=np.array([env_idx]),
            )  # (1, 6) rad
            q_np = q.cpu().numpy()[0]   # (6,)
            q_np[5] = np.deg2rad(gripper_deg)
            return rad2deg_batch(q_np)  # (6,) deg
        except Exception:
            return None

    def plan(self, env_idx: int = 0) -> list[np.ndarray]:
        """
        返回完整轨迹：每个元素是 (6,) 目标角度（度）。
        失败时返回空列表。
        """
        traj = []
        cube = self.cube_pos
        down_quat = np.array([0.0, 1.0, 0.0, 0.0])  # 末端朝下

        # 1. Home
        home = HOME_DEG.copy()
        traj += interpolate_joints(HOME_DEG, home, n_steps=20)

        # 2. Pre-grasp（方块正上方 10 cm）
        pregrasp_pos = cube + np.array([0.0, 0.0, 0.10])
        q_pre = self.solve_ik_deg(pregrasp_pos, down_quat, GRIPPER_OPEN_DEG, env_idx)
        if q_pre is None:
            return []
        traj += interpolate_joints(home, q_pre, n_steps=40)

        # 3. Approach（靠近方块 2 cm）
        approach_pos = cube + np.array([0.0, 0.0, 0.02])
        q_app = self.solve_ik_deg(approach_pos, down_quat, GRIPPER_OPEN_DEG, env_idx)
        if q_app is None:
            return []
        traj += interpolate_joints(q_pre, q_app, n_steps=30)

        # 4. Grasp（闭合夹爪，保持位置）
        q_grasp = q_app.copy()
        q_grasp[5] = GRIPPER_CLOSED_DEG
        traj += interpolate_joints(q_app, q_grasp, n_steps=20)

        # 5. Lift（抬高 15 cm）
        lift_pos = cube + np.array([0.0, 0.0, 0.15])
        q_lift = self.solve_ik_deg(lift_pos, down_quat, GRIPPER_CLOSED_DEG, env_idx)
        if q_lift is None:
            return []
        traj += interpolate_joints(q_grasp, q_lift, n_steps=40)

        # 6. Place（移动到放置区域：右侧 10 cm 偏移）
        place_pos = cube + np.array([0.10, 0.10, 0.06])
        q_place = self.solve_ik_deg(place_pos, down_quat, GRIPPER_CLOSED_DEG, env_idx)
        if q_place is None:
            return []
        traj += interpolate_joints(q_lift, q_place, n_steps=50)

        # 7. Release（张开夹爪）
        q_release = q_place.copy()
        q_release[5] = GRIPPER_OPEN_DEG
        traj += interpolate_joints(q_place, q_release, n_steps=15)

        # 8. Return to Home
        traj += interpolate_joints(q_release, HOME_DEG, n_steps=40)

        return traj


# ── 主采集循环 ────────────────────────────────────────────────────────────────

def build_scene(cfg: SDGConfig):
    """构建 Genesis 场景，返回 (scene, so101, cubes, cam_top, cam_wrist)"""

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=cfg.ctrl_dt, substeps=cfg.substeps),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            enable_joint_limit=True,
            constraint_solver=gs.constraint_solver.Newton,
        ),
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=list(range(min(4, cfg.n_envs))),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.6, -0.5, 0.5),
            camera_lookat=(0.15, 0.0, 0.1),
            camera_fov=45,
        ),
        show_viewer=cfg.show_viewer,
    )

    # ── 地面 ──────────────────────────────────────────────────────────────
    scene.add_entity(gs.morphs.Plane())

    # ── SO-101 ────────────────────────────────────────────────────────────
    so101 = scene.add_entity(
        gs.morphs.MJCF(
            file=cfg.xml_path,
            pos=(0.0, 0.0, 0.0),
        )
    )

    # ── 目标方块（每个 env 一个，域随机化在 build 后设置） ─────────────────
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.03, 0.03, 0.03),
            pos=(0.15, 0.0, 0.015),
        ),
        surface=gs.surfaces.Default(color=(1.0, 0.3, 0.3, 1.0)),
    )

    # ── 相机：俯视 ────────────────────────────────────────────────────────
    cam_top = scene.add_camera(
        res=(cfg.img_width, cfg.img_height),
        pos=(0.0, 0.0, 0.7),
        lookat=(0.15, 0.0, 0.0),
        fov=55,
        GUI=False,
    )

    # ── 相机：侧视 ────────────────────────────────────────────────────────
    cam_side = scene.add_camera(
        res=(cfg.img_width, cfg.img_height),
        pos=(0.5, -0.4, 0.3),
        lookat=(0.15, 0.0, 0.1),
        fov=45,
        GUI=False,
    )

    # ── 构建并行环境 ──────────────────────────────────────────────────────
    scene.build(n_envs=cfg.n_envs)

    # ── PD 增益（build 之后设置） ──────────────────────────────────────────
    kp = np.array([100.0, 100.0, 80.0, 80.0, 60.0, 40.0])
    kv = np.array([10.0,  10.0,   8.0,  8.0,  6.0,  4.0])
    so101.set_dofs_kp(kp, ALL_DOF_IDX)
    so101.set_dofs_kv(kv, ALL_DOF_IDX)

    return scene, so101, cube, cam_top, cam_side


def randomize_env(
    cfg:    SDGConfig,
    cube:   gs.Entity,
    n_envs: int,
) -> np.ndarray:
    """
    随机化各 env 中方块的位置，返回 (n_envs, 3) 世界坐标。
    """
    x = np.random.uniform(*cfg.cube_x_range, size=n_envs)
    y = np.random.uniform(*cfg.cube_y_range, size=n_envs)
    z = np.full(n_envs, 0.015)
    pos = np.stack([x, y, z], axis=-1)   # (N, 3)

    cube_pos_tensor = torch.tensor(pos, dtype=torch.float32, device=gs.device)
    cube.set_pos(cube_pos_tensor)

    return pos   # (N, 3) numpy，供 IK 规划用


def collect_episode(
    cfg:      SDGConfig,
    scene:    gs.Scene,
    so101:    gs.Entity,
    cube:     gs.Entity,
    cam_top:  gs.Camera,
    cam_side: gs.Camera,
    dataset:  LeRobotDataset,
    env_idx:  int,
    cube_pos: np.ndarray,   # (3,)
    task_str: str,
) -> bool:
    """
    采集单个 episode（env_idx 对应的并行环境）。
    返回是否成功（False 表示 IK 规划失败）。
    """
    ee_link = so101.get_link("gripper_link")   # 根据实际 XML 调整

    # 重置该 env 的机器人到 home 姿态
    home_rad = deg2rad_batch(HOME_DEG)
    so101.set_qpos(
        np.tile(home_rad, (1, 1)),
        envs_idx=np.array([env_idx]),
    )
    for _ in range(30):
        scene.step()

    # 规划轨迹
    sm = PickPlaceStateMachine(so101, ee_link, cube_pos)
    trajectory_deg = sm.plan(env_idx=env_idx)
    if not trajectory_deg:
        return False

    # 执行并记录
    for target_deg in trajectory_deg:
        target_rad = deg2rad_batch(target_deg)

        # 控制（并行环境只控制 env_idx）
        so101.control_dofs_position(
            target_rad[np.newaxis, :],   # (1, 6)
            ALL_DOF_IDX,
            envs_idx=np.array([env_idx]),
        )
        scene.step()

        # 读取实际状态（弧度 → 度）
        cur_rad = so101.get_dofs_position(ALL_DOF_IDX, envs_idx=np.array([env_idx]))
        cur_deg = rad2deg_batch(cur_rad)[0]   # (6,)

        # 渲染图像（批量，取 env_idx 这一行）
        rgb_top,  _, _, _ = cam_top.render(rgb=True, depth=False,
                                           segmentation=False, normal=False)
        rgb_side, _, _, _ = cam_side.render(rgb=True, depth=False,
                                            segmentation=False, normal=False)

        # rgb 形状：(n_envs, H, W, 3) 或 (H, W, 3)
        if rgb_top.ndim == 4:
            img_top  = rgb_top[env_idx].cpu().numpy().astype(np.uint8)
            img_side = rgb_side[env_idx].cpu().numpy().astype(np.uint8)
        else:
            img_top  = rgb_top.cpu().numpy().astype(np.uint8)
            img_side = rgb_side.cpu().numpy().astype(np.uint8)

        # 写入 LeRobot 格式
        dataset.add_frame({
            "observation.state":       cur_deg.astype(np.float32),
            "action":                  target_deg.astype(np.float32),
            "observation.images.top":  img_top,
            "observation.images.side": img_side,
            "task":                    task_str,
        })

    dataset.save_episode(task=task_str)
    return True


def create_lerobot_dataset(cfg: SDGConfig) -> LeRobotDataset:
    """创建 LeRobot 数据集（若已存在则追加）"""
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": MOTOR_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": MOTOR_NAMES,
        },
        "observation.images.top": {
            "dtype": "video",
            "shape": (3, 480, 640),
            "names": ["channel", "height", "width"],
        },
        "observation.images.side": {
            "dtype": "video",
            "shape": (3, 480, 640),
            "names": ["channel", "height", "width"],
        },
        "task": {
            "dtype": "string",
            "shape": (1,),
            "names": None,
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=cfg.repo_id,
        fps=cfg.fps,
        features=features,
        robot_type="so101",
        use_videos=True,
    )
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Genesis SO-101 SDG")
    parser.add_argument("--n_envs",      type=int,   default=64)
    parser.add_argument("--n_episodes",  type=int,   default=200)
    parser.add_argument("--repo_id",     type=str,   default="local/so101-genesis-pickplace")
    parser.add_argument("--xml",         type=str,   default="assets/so101_new_calib.xml")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--vis",         action="store_true")
    args = parser.parse_args()

    cfg = SDGConfig(
        xml_path    = args.xml,
        n_envs      = args.n_envs,
        n_episodes  = args.n_episodes,
        repo_id     = args.repo_id,
        push_to_hub = args.push_to_hub,
        show_viewer = args.vis,
    )

    # ── 初始化 Genesis ────────────────────────────────────────────────────
    gs.init(backend=gs.gpu, logging_level="warning")

    # ── 构建场景 ──────────────────────────────────────────────────────────
    print(f"[SDG] 构建场景：{cfg.n_envs} 个并行环境...")
    scene, so101, cube, cam_top, cam_side = build_scene(cfg)
    print("[SDG] 场景构建完成")

    # ── 创建数据集 ────────────────────────────────────────────────────────
    dataset = create_lerobot_dataset(cfg)

    task_descriptions = [
        "pick up the red cube and place it to the right",
        "grasp the cube and lift it up",
        "pick the block and move it to the target location",
    ]

    # ── 主循环：按批次采集 ────────────────────────────────────────────────
    n_collected   = 0
    n_failed      = 0
    t_start       = time.time()

    while n_collected < cfg.n_episodes:
        # 随机化所有 env 的方块位置
        cube_positions = randomize_env(cfg, cube, cfg.n_envs)  # (N, 3)

        # 让物理稳定
        for _ in range(20):
            scene.step()

        # 轮询每个 env 采集一个 episode
        for env_idx in range(cfg.n_envs):
            if n_collected >= cfg.n_episodes:
                break

            task_str = task_descriptions[n_collected % len(task_descriptions)]
            success = collect_episode(
                cfg       = cfg,
                scene     = scene,
                so101     = so101,
                cube      = cube,
                cam_top   = cam_top,
                cam_side  = cam_side,
                dataset   = dataset,
                env_idx   = env_idx,
                cube_pos  = cube_positions[env_idx],
                task_str  = task_str,
            )

            if success:
                n_collected += 1
            else:
                n_failed += 1

            # 进度日志
            if n_collected % 10 == 0 and n_collected > 0:
                elapsed = time.time() - t_start
                rate = n_collected / elapsed
                eta  = (cfg.n_episodes - n_collected) / max(rate, 1e-6)
                print(
                    f"[SDG] {n_collected}/{cfg.n_episodes} episodes "
                    f"| 失败 {n_failed} "
                    f"| {rate:.1f} ep/s "
                    f"| ETA {eta/60:.1f} min"
                )

    # ── 收尾 ──────────────────────────────────────────────────────────────
    dataset.consolidate(run_compute_stats=True)

    if cfg.push_to_hub:
        print(f"[SDG] 推送到 HuggingFace Hub: {cfg.repo_id}")
        dataset.push_to_hub()

    elapsed = time.time() - t_start
    print(f"\n[SDG] 完成！共采集 {n_collected} episodes，失败 {n_failed}，耗时 {elapsed/60:.1f} min")
    print(f"[SDG] 数据集路径：{dataset.root}")


if __name__ == "__main__":
    main()
```

---

## 5. 关键设计决策

### 5.1 deg ↔ rad 转换

Genesis 物理引擎内部使用弧度，LeRobot 数据集使用角度制（与真实 SO-101 Dynamixel 电机一致）。

```python
# 控制时：度 → 弧度
so101.control_dofs_position(np.deg2rad(target_deg))

# 读取时：弧度 → 度
state_deg = np.rad2deg(so101.get_dofs_position(ALL_DOF_IDX).cpu().numpy())

# 写入数据集时：始终用度
dataset.add_frame({"observation.state": state_deg, "action": target_deg})
```

### 5.2 并行 vs 串行采集策略

Genesis 的并行优势在于**同时 step 所有环境**。但由于每个 episode 的轨迹长度不同（IK 规划结果各异），当前实现采用**串行按环境循环**的方式，简化了数据记录逻辑。

如果追求最大吞吐量，可改为完全并行（所有环境同步执行相同步数的轨迹），但需要处理对齐问题：

```python
# 高性能并行版本框架（伪代码）
trajectories = [sm.plan(i) for i, sm in enumerate(state_machines)]
max_len = max(len(t) for t in trajectories)

for step in range(max_len):
    # 批量控制（所有 env 同时）
    targets_rad = np.array([
        deg2rad_batch(traj[min(step, len(traj)-1)])
        for traj in trajectories
    ])  # (N, 6)
    so101.control_dofs_position(targets_rad, ALL_DOF_IDX)
    scene.step()

    # 批量读取图像（一次渲染所有 env）
    rgb_top, _, _, _ = cam_top.render(rgb=True, ...)  # (N, H, W, 3)
    # ...分别写入各 env 的 frame
```

### 5.3 LeRobot Dataset 版本兼容

LeRobot v2 和 v3 的 `add_frame` / `save_episode` API 略有差异：

```python
# v2（旧）
dataset.add_frame(frame_dict)
dataset.save_episode(task=task_str)

# v3（新，推荐）
dataset.add_frame(frame_dict)
dataset.save_episode(task=task_str)
dataset.consolidate(run_compute_stats=True)   # 最终需调用一次
```

---

## 6. 已知问题与规避

### 6.1 SO-101 XML 链接名称

Genesis 加载 MJCF 后，链接名称来自 XML 文件。需要根据实际 `so101_new_calib.xml` 确认末端执行器链接名：

```python
# 查看所有链接名称
for link in so101.links:
    print(link.name)
# 常见：'gripper_link', 'Fixed_Jaw', 'wrist_link' 等
```

### 6.2 MPM 软体物体无法抓取

Genesis Issue [#1858](https://github.com/Genesis-Embodied-AI/Genesis/issues/1858) 报告：MPM（弹性/可变形）物体目前无法被刚性夹爪正确抓取。

**规避方案**：当前版本使用 `gs.morphs.Box`（刚性方块），软体物体任务等待 Genesis 修复后再启用。

### 6.3 大批量渲染显存

当 `n_envs=256` 且分辨率 640×480 时，批量渲染会占用大量显存。

**规避方案**：

```python
# 方案 A：降低渲染分辨率
cam_top = scene.add_camera(res=(320, 240), ...)

# 方案 B：使用 BatchRenderer（更高效）
scene = gs.Scene(
    renderer=gs.options.renderers.BatchRenderer(use_rasterizer=True),
    ...
)

# 方案 C：分批渲染（每次只渲染部分 env）
# 在 vis_options 中指定 rendered_envs_idx
```

### 6.4 IK 收敛失败

目标位置超出机器人工作空间时 IK 会失败。

**规避方案**：

```python
# 增加重试逻辑
for _ in range(3):
    q = so101.inverse_kinematics(link=ee_link, pos=pos, quat=quat)
    if q is not None and not torch.isnan(q).any():
        break
    pos += np.random.randn(3) * 0.005   # 微扰
```

---

## 7. 落地步骤

```
Week 1：环境搭建 + 单 env 验证
├── pip install genesis-world lerobot
├── 确认 so101_new_calib.xml 路径和链接名称
├── 运行 validate_so101_genesis.py --vis
└── 对比输出的 state/action 分布与真实数据集

Week 2：IK + 状态机调试
├── 调整 HOME_DEG 和各阶段 IK 目标位置
├── 调整 PD 增益（kp, kv）使关节跟踪平滑
├── 验证夹爪开合角度（GRIPPER_OPEN/CLOSED_DEG）
└── 单 episode 成功率 > 90%

Week 3：并行扩展 + 域随机化
├── 启用 n_envs=64，验证稳定性
├── 添加物体颜色/质量随机化
├── 批量生成 100 episodes，验证数据格式
└── 推送测试数据集到 HuggingFace

Week 4：大规模生产
├── n_envs=256，生成 1000+ episodes
├── 添加光照随机化（scene.add_light）
├── 与真实数据集混合训练 SmolVLA
└── 验证 sim-to-real 迁移效果
```

---

## 参考资料

- [Genesis 文档](https://genesis-world.readthedocs.io/)
- [Genesis examples/manipulation/grasp_env.py](https://github.com/Genesis-Embodied-AI/Genesis/blob/main/examples/manipulation/grasp_env.py)
- [Genesis examples/tutorials/IK_motion_planning_grasp.py](https://github.com/Genesis-Embodied-AI/Genesis/blob/main/examples/tutorials/IK_motion_planning_grasp.py)
- [Genesis Issue #1858：SO-101 + Genesis + LeRobot](https://github.com/Genesis-Embodied-AI/Genesis/issues/1858)
- [LeRobot Dataset Format](https://github.com/huggingface/lerobot/blob/main/docs/datasets.md)
- [robosim-so101-pickup-v2（MuJoCo 版参考数据集）](https://huggingface.co/datasets/houmans/robosim-so101-pickup-v2)
- [so101-autogen（Isaac Sim 版参考实现）](https://github.com/haoran1062/so101-autogen)
