# Genesis × SO-101 合成数据生成 — 测试报告

> 测试环境：NVIDIA RTX 4090 (david@10.161.176.110)
> Docker 镜像：genesis_poc:v2（基于 pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel + genesis-world + libxrender1 + xvfb）
> 日期：2026-03-05

---

## 一、测试概览

| 脚本 | 定位 | 结果 |
|------|------|------|
| `poc_genesis_pipeline.py` | 数据管线可行性验证（Franka fallback） | **8/8 全部通过** |
| `sdg_so101_genesis.py` | SO-101 URDF 真机模型 SDG 验证 | **7/7 全部通过** |

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
  python -u /workspace/lfzte/02_intermediate/scripts/poc_genesis_pipeline.py \
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
  python -u /workspace/lfzte/02_intermediate/scripts/sdg_so101_genesis.py \
    --episodes 3 --fps 30 --save /tmp/sdg_so101_output
```

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
- SO-101 URDF 6-DOF 关节正确识别，IK 求解可用
- 双相机 (480×640) 渲染正常
- 输出数据结构与 svla_so101_pickplace 完全兼容
- 单 episode（240 帧 × 双相机渲染）耗时约 100s
- **LeRobot v3 格式打包完成**：Parquet + MP4 + meta/info.json
- 合成数据已回传本地

### 待完成

1. **并行环境加速**：利用 Genesis `n_envs` 参数并行化，预计 10-50x 加速
2. **轨迹质量优化**：调优 PD 增益（当前 Home 误差 8.27°），改进 IK 求解策略
3. **域随机化扩展**：增加光照、纹理、物体类型等随机化维度
4. **大规模采集**：目标 50-500 episodes，对齐 svla_so101_pickplace 规模
5. **HuggingFace Hub 推送**：使用 LeRobot API 直接推送到 Hub
