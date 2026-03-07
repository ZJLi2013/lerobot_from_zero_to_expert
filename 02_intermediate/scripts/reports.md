# Genesis × SO-101 合成数据生成 — 测试报告

> 测试环境：NVIDIA RTX 4090
> Docker 镜像：genesis_poc:latest / v2（基于 pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel + genesis-world + libxrender1 + xvfb）
> 日期：2026-03-05 ~ 2026-03-06

---

## 一、测试概览

| 脚本 | 定位 | 结果 | 日期 |
|------|------|------|------|
| `1_poc_pipeline.py` | 数据管线可行性验证（Franka fallback） | **8/8 全部通过** | 03-05 |
| `2_collect.py` (URDF) | SO-101 URDF 真机模型 SDG 验证 | **7/7 全部通过** | 03-05 |
| `2_collect.py` (MJCF) | SO-101 官方 MJCF 采集验证 | **7/7 全部通过** | 03-06 |

---

## 二、POC：数据管线可行性验证

### 2.1 目标

验证 Genesis 物理引擎在 RTX 4090 + Docker + headless 环境下的完整数据管线：
GPU 初始化 → 场景构建 → 物理步进 → 相机渲染 → 机器人加载 → IK 求解 → 关节控制 → 帧采集保存。

### 2.2 测试阶段与结果

| 阶段 | 验证内容 | 结果 | 关键数据 |
|------|---------|------|---------|
| 1/8 | Genesis import + GPU init | ✓ PASS | torch 2.8.0+cu126, RTX 4090 |
| 2/8 | 场景构建 (Plane + Box + Camera) | ✓ PASS | 320×240 |
| 3/8 | 物理步进 + 相机渲染 | ✓ PASS | 10 步 = 21ms |
| 4/8 | 机器人链接 & DOF 探测 | ✓ PASS | 9 DOF (Franka Panda) |
| 5/8 | PD 增益 + Home 姿态 | ✓ PASS | — |
| 6/8 | IK 求解 | ✓ PASS | 3415ms |
| 7/8 | 关节控制 + 状态读取 | ✓ PASS | tracking err 40.6° |
| 8/8 | 采集 10 帧 → .npy | ✓ PASS | states(10,9), images(10,240,320,3) |

### 2.3 关键发现

1. **Headless 渲染**：必须启动 Xvfb 虚拟显示（脚本内 `ensure_display()` 自动处理）。
2. **Docker volume mount**：`genesis_poc` 镜像由 `docker commit` 产生，`/workspace` 已含旧数据，需注意路径冲突。直接用 `docker run ... python /path/to/script.py` 避免 `bash -c` 的 CRLF 和路径问题。
3. **SO-101 XML 缺失**时自动 fallback 到 Franka Panda，不影响管线验证。

### 2.4 运行命令

```bash
docker run --rm --gpus all \
  -v ~/github/lerobot_from_zero_to_expert:/workspace/lfzte \
  genesis_poc:v2 \
  python -u /workspace/lfzte/02_intermediate/scripts/genesis_quick_start.py \
    --frames 10 --save /tmp/poc_output
```

---

## 三、SDG：SO-101 URDF 合成数据生成

### 3.1 目标

使用真实 SO-101 URDF 模型（[haixuantao/dora-bambot](https://huggingface.co/haixuantao/dora-bambot/blob/main/URDF/so101.urdf)），
在 Genesis 中完成完整的合成数据采集，验证输出数据结构与 `svla_so101_pickplace` 完全兼容。

### 3.2 测试阶段与结果

| 阶段 | 验证内容 | 结果 | 关键数据 |
|------|---------|------|---------|
| 1/7 | SO-101 URDF 下载 | ✓ PASS | 1 URDF + 13 STL meshes |
| 2/7 | Genesis + 场景构建 | ✓ PASS | 30Hz, 双相机 640×480 |
| 3/7 | 关节发现 & 语义映射 | ✓ PASS | **6 DOF** (正确) |
| 4/7 | PD 增益 + Home 姿态 | ✓ PASS | mean_err=8.27° |
| 5/7 | IK 求解 | ✓ PASS | 3312ms |
| 6/7 | 采集 3 episodes | ✓ PASS | 720 frames, 307s |
| 7/7 | 保存 + 统计 | ✓ PASS | 见下表 |

### 3.3 svla_so101_pickplace 兼容性

| 维度 | SDG 输出 | 目标值 | 匹配 |
|------|---------|--------|------|
| state dim | 6 | 6 | ✓ |
| action dim | 6 | 6 | ✓ |
| image shape | (480, 640, 3) | (480, 640, 3) | ✓ |
| cameras | up + side | up + side | ✓ |
| fps | 30 | 30 | ✓ |
| 关节映射 | shoulder_pan → gripper | shoulder_pan → gripper | ✓ |

**结论：数据结构与 svla_so101_pickplace 完全兼容。**

### 3.4 输出数据统计

```
states:      (720, 6), range=[-147.6, 100.0]°
actions:     (720, 6), range=[-147.6, 100.0]°
images_up:   (720, 480, 640, 3), range=[21, 255]
images_side: (720, 480, 640, 3), range=[10, 255]
```

### 3.5 关键设计决策

1. **URDF `fixed=True`**：Genesis 加载 URDF 时默认添加 6-DOF root_joint（共 12 DOF）。必须设置 `fixed=True` 固定基座，使 DOF 数正确为 6。
2. **自动下载 URDF + STL**：脚本从 HuggingFace 自动下载 SO-101 URDF 和 13 个 STL mesh 文件，无需手动准备资源。
3. **IK-based scripted trajectory**：使用 Genesis 内置 IK 求解器规划 pick-place 轨迹（Home → Pre-grasp → Approach → Grasp → Lift → Return）。
4. **域随机化**：每个 episode 随机化方块位置 (x∈[0.10,0.22], y∈[-0.08,0.08])。

### 3.6 运行命令

```bash
docker run --rm --gpus all \
  -v ~/github/lerobot_from_zero_to_expert:/workspace/lfzte \
  genesis_poc:v2 \
  python -u /workspace/lfzte/02_intermediate/scripts/2_collect.py \
    --episodes 3 --fps 30 --save /tmp/sdg_so101_output
```

---

## 三-B、SDG：SO-101 MJCF 官方模型采集（2026-03-06）

### 3B.1 目标

将 `2_collect.py` 从 URDF 切换为 **LeRobot 官方 SO-101 MJCF** (`so101_new_calib.xml`)，
验证 MJCF 加载、关节发现、IK 求解、多 episode 采集的完整链路。

### 3B.2 MJCF 来源

`so101_new_calib.xml` 托管于 HuggingFace 数据集 `Genesis-Intelligence/assets`，
脚本通过 `huggingface_hub.snapshot_download` 自动下载 `SO101/*`（含 XML + STL mesh）。

Docker 运行时需额外安装 `huggingface_hub`：

```bash
docker run --rm --gpus all \
  -v ~/github/lerobot_from_zero_to_expert:/workspace/lfzte \
  -v ~/sdg_collect_output:/output \
  genesis_poc:latest \
  bash -c 'pip install -q huggingface_hub && \
    python -u /workspace/lfzte/02_intermediate/scripts/2_collect.py \
      --episodes 3 --episode-length 8 --save /output'
```

### 3B.3 测试阶段与结果

| 阶段 | 验证内容 | 结果 | 关键数据 |
|------|---------|------|---------|
| 1/7 | SO-101 MJCF 定位 + HF 下载 | ✓ PASS | 33 files, ~55s 下载 |
| 2/7 | Genesis + 场景构建 | ✓ PASS | GPU: RTX 4090, 30Hz, 双相机 640×480 |
| 3/7 | 关节发现 & EE link | ✓ PASS | **6 DOF**, EE=`gripper` |
| 4/7 | PD 增益 + Home 姿态 | ✓ PASS | **mean_err=0.12°**（URDF 版 8.27°） |
| 5/7 | 数据采集 × 3 episodes | ✓ PASS | 720 frames, 286.9s |
| 6/7 | 保存 npy | ✓ PASS | 见下表 |
| 7/7 | 生成 .rrd | ✓ PASS | 128.9 MB |

### 3B.4 MJCF vs URDF 对比

| 指标 | URDF (03-05) | MJCF (03-06) | 改进 |
|------|-------------|-------------|------|
| 模型来源 | HuggingFace URDF + 13 STL 手动下载 | Genesis-Intelligence/assets 自动下载 | 更简洁 |
| Home 跟踪误差 | 8.27° | **0.12°** | **69× 改善** |
| DOF 数 | 6（需 `fixed=True`） | 6（MJCF 天然固定基座） | 无需额外参数 |
| EE link | `Fixed_Jaw` / `gripper` | `gripper` | 一致 |
| 加载复杂度 | URDF 下载 + STL + `fixed=True` + base_offset | MJCF 单文件 + auto-download | 大幅简化 |
| 代码行数 | ~650 行 | ~430 行 | -34% |
| state range | [-147.6°, 100.0°] | [-154.3°, 121.5°] | IK 覆盖范围更大 |
| 采集速度 | ~100s/episode | ~96s/episode | 相当 |

### 3B.5 输出数据统计

```
states:      (720, 6), range=[-154.3, 121.5]°
actions:     (720, 6), range=[-154.3, 121.6]°
images_up:   (720, 480, 640, 3)
images_side: (720, 480, 640, 3)
.rrd:        128.9 MB
```

### 3B.6 关键参数

```
MJCF     = Genesis-Intelligence/assets → SO101/so101_new_calib.xml
HOME°    = [0.0, -30.0, 90.0, -60.0, 0.0, 0.0]
IK quat  = [0.0, 1.0, 0.0, 0.0] (gripper-down)
PD gains = kp=[500, 500, 400, 400, 300, 200], kv=[50, 50, 40, 40, 30, 20]
```

### 3B.7 结论

MJCF 版本相比 URDF 版本在 Home 跟踪精度上改善 69 倍（8.27° → 0.12°），
代码复杂度降低 34%，且模型加载流程完全自动化。**建议后续统一使用 MJCF 路线。**

已回传本地：`02_intermediate/sdg_data/sdg_collect_mjcf.rrd`

---

## 四、环境搭建流程

### 4.1 Docker 镜像构建

```bash
# 基础镜像
docker run -d --name poc_setup --gpus all pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel tail -f /dev/null

# 安装依赖
docker exec poc_setup apt-get update -q
docker exec poc_setup apt-get install -y xvfb libgl1 libgles2 libegl1 libglx-mesa0 libxrender1
docker exec poc_setup pip install genesis-world

# 提交镜像
docker commit poc_setup genesis_poc:latest

# 清理
docker stop poc_setup && docker rm poc_setup
```

### 4.2 Local Push → Remote Pull 流程

```bash
# 本地（Windows）
git add . && git commit -m "..." && git push origin main

# 远端（Linux 4090）
cd ~/github/lerobot_from_zero_to_expert && git pull

# 运行（Docker，注意 python -u 避免输出缓冲）
docker run --rm --gpus all \
  -v ~/github/lerobot_from_zero_to_expert:/workspace/lfzte \
  genesis_poc:v2 \
  python -u /workspace/lfzte/02_intermediate/scripts/<script>.py
```

### 4.3 踩坑记录

| 问题 | 原因 | 解法 |
|------|------|------|
| CRLF 导致 bash 脚本崩溃 | Windows 写的 .sh 在 Linux 解析失败 | 直接用 `python` 命令，不用 `bash -c` 包装 |
| `XRenderFindVisualFormat` 错误 | 缺少 `libxrender1` | `apt install libxrender1` |
| `DISPLAY` 未设置 | Headless GPU 服务器 | 脚本自动启动 Xvfb :99 |
| URDF 12 DOF（预期 6） | Genesis 添加 free root_joint | `gs.morphs.URDF(fixed=True)` |
| Docker volume mount 被遮蔽 | `docker commit` 保留了旧 workdir 内容 | 挂载到不冲突路径或直接用绝对路径 |
| 输出 15 行后不再更新 | Python stdout 管道缓冲 | 使用 `python -u` 禁用缓冲 |

---

## 五、LeRobot 格式打包

### 5.1 转换脚本

`npy_to_lerobot.py` 将 SDG 输出的 npy 文件转为 LeRobot v3 标准格式。

```bash
docker run --rm \
  -v ~/github/lerobot_from_zero_to_expert:/workspace/lfzte \
  -v ~/sdg_output:/output \
  genesis_poc:v3 \
  python -u /workspace/lfzte/02_intermediate/scripts/npy_to_lerobot.py \
    --input /output/npy --output /output/lerobot_dataset --fps 30
```

### 5.2 输出结构

```
lerobot_dataset/
  data/chunk-000/file-000.parquet          (40.5 KB, 720 rows)
  videos/observation.images.up/chunk-000/
    episode_000000.mp4                     (795.0 KB, 240 frames)
    episode_000001.mp4                     (852.5 KB)
    episode_000002.mp4                     (701.2 KB)
  videos/observation.images.side/chunk-000/
    episode_000000.mp4                     (1189.0 KB, 240 frames)
    episode_000001.mp4                     (1282.8 KB)
    episode_000002.mp4                     (1100.5 KB)
  meta/
    info.json                              (2.2 KB)
    episodes/chunk-000/file-000.parquet    (2.9 KB)
    tasks/chunk-000/file-000.parquet       (1.9 KB)
```

### 5.3 info.json 关键字段

| 字段 | 值 |
|------|-----|
| codebase_version | v3.0 |
| robot_type | so101 |
| total_episodes | 3 |
| total_frames | 720 |
| fps | 30 |
| observation.state | float32, shape=[6], names=[shoulder_pan...gripper] |
| action | float32, shape=[6] |
| observation.images.up | video, [3, 480, 640], mp4v |
| observation.images.side | video, [3, 480, 640], mp4v |

### 5.4 数据已回传本地

```
02_intermediate/sdg_data/
  lerobot_dataset/   ← LeRobot v3 格式（Parquet + MP4 + meta）
  npy/               ← 原始 npy 文件（states, actions, images）
```

---

## 六、Docker 镜像版本

| 镜像 | 新增内容 | 用途 |
|------|---------|------|
| genesis_poc:latest | genesis-world + OpenGL libs + xvfb | POC 验证 |
| genesis_poc:v2 | + libxrender1 | SDG 运行 |
| genesis_poc:v3 | + pyarrow + opencv-headless + pandas | LeRobot 格式转换 |

---

## 七、结论与下一步

### 已验证

- Genesis 在 RTX 4090 Docker 环境下可稳定运行
- SO-101 **MJCF（官方 MJCF）** 6-DOF 关节正确识别，Home 跟踪误差 0.12°
- SO-101 URDF 6-DOF 亦可工作（Home 误差 8.27°，已被 MJCF 取代）
- 双相机 (480×640) 渲染正常
- 输出数据结构与 svla_so101_pickplace 完全兼容
- 单 episode（240 帧 × 双相机渲染）耗时约 96s
- **LeRobot v3 格式打包完成**：Parquet + MP4 + meta/info.json
- 合成数据已回传本地（`.rrd` + `.npy`）

### 待完成

1. **并行环境加速**：利用 Genesis `n_envs` 参数并行化，预计 10-50x 加速
2. **域随机化扩展**：增加光照、纹理、物体类型等随机化维度
3. **大规模采集**：目标 50-500 episodes，对齐 svla_so101_pickplace 规模
4. **HuggingFace Hub 推送**：使用 LeRobot API 直接推送到 Hub
5. **Docker 镜像优化**：将 `huggingface_hub` 预装入 genesis_poc 镜像

---

## 八、AMD MI308 验证记录（新增）

### 8.1 测试环境

| 项目 | 信息 |
|------|------|
| 远端节点 | 已脱敏 |
| GPU | AMD Instinct MI308X × 8 (`gfx942`) |
| 基础镜像 | `rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0` |
| 运行模式 | **Headless**（脚本内 `Xvfb :99`） |
| 测试脚本 | `02_intermediate/scripts/2_collect.py` |
| 可视化脚本 | `02_intermediate/scripts/viz_sdg_rerun.py` |

### 8.2 兼容性处理

在该 ROCm 镜像内，直接安装 `genesis-world` 后运行会在 `scene.build()` 报二进制兼容错误：

- `ValueError: numpy.dtype size changed, may indicate binary incompatibility`

修复方式：安装依赖时固定 NumPy 到 1.26 系列（已验证 `1.26.4` 可用）。

```bash
python -m pip install --no-input numpy==1.26.4 genesis-world rerun-sdk
```

### 8.3 运行命令（MI308）

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v ~/github/lerobot_from_zero_to_expert:/workspace/lfzte \
  -v ~/sdg_mi308_output:/output \
  rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0 \
  bash -lc '
    apt-get update && apt-get install -y xvfb libgl1 libgles2 libegl1 libglx-mesa0 libxrender1 &&
    python -m pip install --no-input numpy==1.26.4 genesis-world rerun-sdk &&
    python -u /workspace/lfzte/02_intermediate/scripts/2_collect.py --episodes 1 --episode-length 6 --save /output/npy_mi308 &&
    python -u /workspace/lfzte/02_intermediate/scripts/viz_sdg_rerun.py --input /output/npy_mi308 --output /output/rrd_mi308 --episodes 0
  '
```

### 8.4 结果

`2_collect.py` 在 MI308 上 **7/7 全部通过**：

- 1 episode / 180 frames
- `state`: `(180, 6)`, `action`: `(180, 6)`
- 图像：`(180, 480, 640, 3)`（up/side）
- `svla_so101_pickplace` 结构兼容检查通过

`viz_sdg_rerun.py` 生成 `.rrd` 成功：

- 远端：`/output/rrd_mi308/so101_sdg_episode_0.rrd`（约 29.4 MB）
- 本地回传：`02_intermediate/sdg_data/so101_sdg_episode_0_mi308.rrd`

### 8.5 结论

在指定 AMD 基础镜像与 headless 条件下，SO-101 Genesis SDG 管线可稳定运行并产出可视化 `.rrd`。  
当前关键注意事项是固定 `numpy==1.26.4` 以避免 Genesis 二进制 ABI 冲突。

---

## 九、SO-101 抓取实验：结构性阻塞确认（E42–M1, 2026-03-06）

### 9.1 实验目标

在 `grasp_center` IK 精度已验证（0.0002m）的前提下，通过系统性实验确认抓取失败的根因，并尝试多种修复方案。

### 9.2 实验环境

| 项目 | 信息 |
|------|------|
| GPU | NVIDIA GeForce RTX 4090 |
| Docker 镜像 | genesis_poc:latest |
| 基线脚本 | `3_grasp_experiment.py`（E42–E44） |
| 新增诊断脚本 | `diag_link_positions.py`, `diag_gc_sweep.py` |
| 新增测试脚本 | `5_platform_grasp_test.py`, `6_oriented_grasp_test.py`, `7_minimal_grasp.py` |

### 9.3 实验流程与关键结果

| 实验 | 方案 | 核心参数 | 结果 | 关键发现 |
|------|------|---------|------|---------|
| diag_link_positions | 查询各 link 在不同 IK 目标下的世界坐标 | grasp_center → [0.16,0,0.015] | jaw_mid_z=0.097 | **grasp_center 在 gripper body 下方 8cm，夹爪永远碰不到 cube** |
| diag_gc_sweep | 扫描 grasp_center target_z (-0.08~0.115) | 40 个 target_z 值 | 最低 jaw_mid_z=0.073 | **工作空间限制：jaw 无法下探到 z<0.07** |
| XML 修正 | 移动 grasp_center 到 jaw pinch center | pos=(0.01,0.009,-0.012) | IK err=0.077m | 太近 gripper body，IK 无法到达 |
| E42 | grasp_center IK + 小范围 xy 扫描 | approach_z=0.012, gc=-20, 25 trials | 0/1 成功, Δz=0.0003m | jaw 在 cube 上方 7cm |
| P1 sweep | 抬高 cube（platform=6cm）+ approach_z 扫描 | 20 个 approach_z 值 | trial 最佳 +0.0118m | trial 偶发正向 Δz，full episode -0.067m（cube 被撞落） |
| P2 autotune | platform + approach_z=-0.035 + offset 搜索 | 25 trials | trial 最佳 +0.019m | full episode -0.005m（同上） |
| P3 lateral | 侧向 approach 避免碰撞 | lateral_offset_y=0.06 | 全部 Δz≈0.000 | 避免了碰撞，但也无接触 |
| O1 oriented | gripper link + quat=[0,1,0,0] | IK with 朝向约束 | IK err=0.061m | **5-DOF 无法同时满足位置+朝向** |
| O2 tall_platform | platform=12cm + gripper link | 高台 + height sweep | cube 坠落 | 平台太窄不稳定 |
| E43-E44 | 原始脚本 + 快速 episode(3s) | approach_z=-0.065, 90 steps | 0/1 成功 | 发现 verbose=True bug |
| **M1 consistency** | **独立重复 10 次同一 trial** | offset=(-0.005,+0.005) | **0/10 > 0.01m, mean=+0.0006m** | **trial 正向 Δz 不可复现** |

### 9.4 发现的 Bug

**`3_grasp_experiment.py` verbose 诊断状态泄漏**

`build_trajectory_chained_ik(verbose=True)` 在函数内部执行了 `so101.set_qpos()` + `scene.step()`，把 robot 和 cube 从 home 位姿移到 approach 位姿。紧接着的 full episode 循环继承了被扰乱的初始状态，而 trial 每次从 `reset_scene()` 干净启动。

修复：在 verbose 调用后、episode 收集前加入 `reset_scene(cube_pos, settle_steps=30)`。

### 9.5 根因总结

```
抓取失败
├── 直接原因：夹爪无法接触 cube
│   ├── grasp_center 在 gripper body 下方 ~8cm（MJCF gripperframe 定义错误）
│   │   IK 精确送达 grasp_center，但 jaw 在 cube 上方 7-9cm
│   └── jaw 最低可达高度 z=0.073m，cube top z=0.030m，gap=4.3cm
├── 误导因素：trial/episode gap
│   ├── verbose=True bug 扰乱了 episode 初始状态
│   └── auto-tune 中偶发的正向 Δz 来自仿真状态累积，独立重复 0/10 可复现
└── 结构性限制
    ├── SO-101 仅 5+1 DOF，无法同时约束位置和朝向（quat IK err > 0.06m）
    └── 当前 MJCF 的 link 几何使 jaw 无法下探到桌面高度
```

### 9.6 新增脚本清单

| 脚本 | 用途 | 输出 |
|------|------|------|
| `diag_link_positions.py` | 在 HOME/IK 目标下打印所有 link 世界坐标 | 终端表格 |
| `diag_gc_sweep.py` | 扫描 grasp_center target_z，对照 IK 精度与 jaw 位置 | 终端表格 + 最佳 target_z |
| `5_platform_grasp_test.py` | 平台抬高 cube + 侧向/垂直 approach + auto-tune | metrics.json + debug PNGs |
| `6_oriented_grasp_test.py` | gripper link + 朝向约束 IK | metrics.json + debug PNGs |
| `7_minimal_grasp.py` | 最小化 trial 一致性测试（N 次独立重复） | metrics.json + success rate |

### 9.7 结论（已修正，基于 debug PNG 复查）

> **之前"z 方向不可达"的结论是错误的。** `diag_gc_sweep.py` 测量的 `jaw_mid_z`
> 是 link body origin 中点，不是 jaw mesh tip。实际 jaw 尖端可以到达 cube 高度。

回看 E41/E42/E44 的 close 阶段 debug PNGs，确认：

1. **Z 高度没有问题** — jaw 尖端确实触碰到了 cube
2. **XY 偏移是真正根因** — cube 一直在 fixed jaw 的外侧边缘，没有进入两爪之间的 pinch 区域
3. 夹爪在 cube 旁边"空抓"，close 时只是擦边推动 cube

**真正的阻塞是 `grasp_center` 的 XY 偏移未对准 jaw pinch center。** 修复方向：通过精细 XY offset 网格搜索（步长 1-2mm）找到正确的补偿值，然后修正 MJCF 中 `grasp_center` 的 pos。

详细建议见 `best_practices.md` 9.4.5 节。
