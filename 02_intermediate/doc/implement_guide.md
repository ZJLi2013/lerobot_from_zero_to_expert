# Genesis x SO-101 SDG Implementation Guide

> 面向"可跑通、可复现、可调参"的实现文档。

---

## 目录

1. [目标与范围](#1-目标与范围)
2. [项目结构](#2-项目结构)
3. [架构设计](#3-架构设计)
4. [环境准备](#4-环境准备)
5. [脚本说明与运行](#5-脚本说明与运行)
6. [关键设计决策](#6-关键设计决策)
7. [数据输出说明](#7-数据输出说明)
8. [验收标准](#8-验收标准)
9. [已知问题与规避](#9-已知问题与规避)
10. [常见问题](#10-常见问题)
11. [最小命令清单](#11-最小命令清单)
12. [参考资料](#12-参考资料)

---

## 1. 目标与范围

本指南聚焦 3 件事：

1. 在 Genesis 中稳定运行 SO-101 抓取数据采集
2. 输出可分析数据（`npy + .rrd + metrics.json`）
3. 输出可训练数据（LeRobot v3：`parquet + mp4 + meta`）

---

## 2. 项目结构

```text
02_intermediate/
├── readme.md
├── doc/
│   ├── implement_guide.md                 # 本文件
│   ├── best_practices.md                  # 调参与验收手册
│   └── survey.md                          # 调研资料
└── scripts/
    ├── check_deps.py                      # 依赖检查
    ├── 1_poc_pipeline.py                  # Genesis POC 验证管线
    ├── 2_collect.py                       # SO-101 采集（probe + .rrd 输出）
    ├── 3_grasp_experiment.py              # 抓取调参实验(推荐)，自动 offset 搜索 + metrics.json
    ├── 4_parallel_lerobot.py              # 并行批量采集（N_ENVS 并行 + 域随机化 + 直接写 LeRobot 格式）
    ├── viz_sdg_rerun.py                   # Rerun 可视化回放
    ├── setup_genesis_env.sh               # Genesis 环境搭建脚本
    ├── run_poc_docker.sh                  # Docker 方式运行 POC
    └── reports.md                         # 实验报告
```

---

## 3. 架构设计

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

### 3.1 关节定义（LeRobot SO-101）

```
                          SO-101  6-DOF 机械臂关节示意图
                              （侧视 · Home 姿态）

          [1] shoulder_lift                    运动类型     角度范围(约)
               ◉─────────────── 大臂 ──┐      ↕ 俯仰      -90° ~ +90°
              ╱                         │
             ╱ [0] shoulder_pan         │
      ↺ 水平旋转                   [2] elbow_flex
            │                           ◉      ↕ 俯仰      -120° ~ +120°
       ┌────┴────┐                      │
       │ ▓▓▓▓▓▓ │                       │
       │  底座   │                  小臂 │
       │ ▓▓▓▓▓▓ │                       │
       └────────┘                  [3] wrist_flex
      ▀▀▀▀▀▀▀▀▀▀▀▀                     ◉      ↕ 俯仰      -90° ~ +90°
      ═══桌面═══                        │
                                   [4] wrist_roll
                                        ◉      ↻ 自旋      -180° ~ +180°
                                        │
                                   [5] gripper
                                      ┌─◉─┐
                                      │   │    ↔ 开合      0° ~ 70°
                                      ╘═══╛

    关节编号    名称              LeRobot motor name       控制单位
    ─────────────────────────────────────────────────────────────
      [0]     shoulder_pan       shoulder_pan.pos          度(°)
      [1]     shoulder_lift      shoulder_lift.pos         度(°)
      [2]     elbow_flex         elbow_flex.pos            度(°)
      [3]     wrist_flex         wrist_flex.pos            度(°)
      [4]     wrist_roll         wrist_roll.pos            度(°)
      [5]     gripper            gripper.pos               度(°)
```

```python
JOINT_NAMES = [
    "shoulder_pan",   # 0 — 底座水平旋转
    "shoulder_lift",  # 1 — 大臂俯仰
    "elbow_flex",     # 2 — 肘部弯曲
    "wrist_flex",     # 3 — 手腕俯仰
    "wrist_roll",     # 4 — 手腕自旋
    "gripper",        # 5 — 夹爪开合 (0=张开, max=闭合)
]
```

---

## 4. 环境准备

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

## 5. 脚本说明与运行

脚本按编号递进，从环境验证到生产级采集：

| 编号 | 脚本 | 作用 | 输出 |
|------|------|------|------|
| 1 | `1_poc_pipeline.py` | Genesis 环境 POC 验证 | 控制台日志 |
| 2 | `2_collect.py` | SO-101 采集（probe + 朝向修正） | `.rrd + npy` |
| 3 | `3_grasp_experiment.py` | 抓取调参实验（推荐入口） | `.rrd + metrics.json` |
| 4 | `4_parallel_lerobot.py` | N_ENVS 并行 + 域随机化 + LeRobot 格式 | LeRobot v3 Dataset |

### 5.1 推荐运行方式（Docker + 3_grasp_experiment）

```bash
docker run --rm --gpus all \
  -v ~/github/lerobot_from_zero_to_expert:/workspace/lfzte \
  -v ~/sdg_grasp_exp:/output \
  genesis_poc:latest \
  python -u /workspace/lfzte/02_intermediate/scripts/3_grasp_experiment.py \
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

### 5.2 并行批量采集（4_parallel_lerobot）

**[4_parallel_lerobot.py](../scripts/4_parallel_lerobot.py)**

核心组件：

- `SDGConfig` — 仿真 / 采集 / 域随机化 / 渲染参数（dataclass）
- `PickPlaceStateMachine` — 基于 IK 的 pick-place 状态机（Home→PreGrasp→Approach→Grasp→Lift→Place→Return）
- `build_scene()` — 构建 Genesis 场景（SO-101 + 方块 + 双相机 + 并行环境）
- `randomize_env()` — 域随机化（方块位置）
- `collect_episode()` — 单 episode 采集循环（控制 → 读取 → 渲染 → 写入 LeRobot）
- `create_lerobot_dataset()` — 创建 LeRobot v3 数据集

```bash
python scripts/4_parallel_lerobot.py \
    --n_envs 64 \
    --n_episodes 500 \
    --repo_id your_hf_username/so101-genesis-pickplace \
    --push_to_hub
```

---

## 6. 关键设计决策

### 6.1 deg ↔ rad 转换

Genesis 物理引擎内部使用弧度，LeRobot 数据集使用角度制（与真实 SO-101 Dynamixel 电机一致）。

```python
# 控制时：度 → 弧度
so101.control_dofs_position(np.deg2rad(target_deg))

# 读取时：弧度 → 度
state_deg = np.rad2deg(so101.get_dofs_position(ALL_DOF_IDX).cpu().numpy())

# 写入数据集时：始终用度
dataset.add_frame({"observation.state": state_deg, "action": target_deg})
```

### 6.2 并行 vs 串行采集策略

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

### 6.3 LeRobot Dataset v3 版本

```python
# v3（推荐）
dataset.add_frame(frame_dict)
dataset.save_episode(task=task_str)
dataset.consolidate(run_compute_stats=True)   # 最终需调用一次
```

---

## 7. 数据输出说明

- `sdg_<exp_id>.rrd`：可视化回放
- `metrics.json`：抓取判定与参数记录
- `states/actions/images_*.npy`：原始数值与图像

---

## 8. 验收标准

以 `metrics.json` 为主：

- `grasp_success == 1`
- `cube_lift_delta > 0.01m`

---

## 9. 已知问题与规避

### 9.1 SO-101 XML 链接名称

Genesis 加载 MJCF 后，链接名称来自 XML 文件。需要根据实际 `so101_new_calib.xml` 确认末端执行器链接名：

```python
# 查看所有链接名称
for link in so101.links:
    print(link.name)
# 常见：'gripper_link', 'Fixed_Jaw', 'wrist_link' 等
```

### 9.2 大批量渲染显存

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

### 9.3 IK 收敛失败

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

## 10. 常见问题

- **Q: 朝向对了但抓不到？**
  A: 先调 `offset_y`，再调 `offset_x`，最后调 `close/hold`。

- **Q: 自动 offset 仍失败？**
  A: 优先确认 `ee_link` 是否真的使用 `gripperframe`；再扩大 offset 搜索范围并联合搜索 `close/hold`。

- **Q: 只想快速验证链路？**
  A: 先跑 `1_poc_pipeline.py`，确认环境无误后再跑抓取实验。

---

## 11. 最小命令清单


### 11.1 4090 拉取最新代码

```bash
ssh david@<4090_HOST> "cd ~/github/lerobot_from_zero_to_expert && git pull origin main"
```

### 11.2 跑 1 episode（自动 offset 调参）

```bash
ssh david@<4090_HOST> "mkdir -p ~/sdg_grasp_exp && docker run --rm --gpus all \
  -v ~/github/lerobot_from_zero_to_expert:/workspace/lfzte \
  -v ~/sdg_grasp_exp:/output \
  genesis_poc:latest \
  python -u /workspace/lfzte/02_intermediate/scripts/3_grasp_experiment.py \
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

### 11.3 回传结果到本地

```powershell
scp david@<4090_HOST>:~/sdg_grasp_exp/E3_auto_offset/improved_sdg_E3_auto_offset.rrd `
  "c:\Users\zhengjli\Documents\github\lerobot_from_zero_to_expert\02_intermediate\sdg_data\improved_sdg_E3_auto_offset.rrd"

scp david@<4090_HOST>:~/sdg_grasp_exp/E3_auto_offset/metrics.json `
  "c:\Users\zhengjli\Documents\github\lerobot_from_zero_to_expert\02_intermediate\sdg_data\metrics_E3_auto_offset.json"
```

### 11.4 本地快速查看结果

```powershell
rerun "c:\Users\zhengjli\Documents\github\lerobot_from_zero_to_expert\02_intermediate\sdg_data\improved_sdg_E3_auto_offset.rrd"
```

判定只看两个值：

- `grasp_success`
- `cube_lift_delta`（目标 > `0.01m`）

---

## 12. 参考资料

- [Genesis 文档](https://genesis-world.readthedocs.io/)
- [Genesis examples/manipulation/grasp_env.py](https://github.com/Genesis-Embodied-AI/Genesis/blob/main/examples/manipulation/grasp_env.py)
- [Genesis examples/tutorials/IK_motion_planning_grasp.py](https://github.com/Genesis-Embodied-AI/Genesis/blob/main/examples/tutorials/IK_motion_planning_grasp.py)
- [Genesis Issue #1858：SO-101 + Genesis + LeRobot](https://github.com/Genesis-Embodied-AI/Genesis/issues/1858)
- [LeRobot Dataset Format](https://github.com/huggingface/lerobot/blob/main/docs/datasets.md)
- [robosim-so101-pickup-v2（MuJoCo 版参考数据集）](https://huggingface.co/datasets/houmans/robosim-so101-pickup-v2)
- [so101-autogen（Isaac Sim 版参考实现）](https://github.com/haoran1062/so101-autogen)
