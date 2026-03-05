# LeRobot 数据集可视化指南

> 数据集: `lerobot/svla_so101_pickplace`  
> 测试节点: `david@10.161.176.110`（远端 headless 服务器，双卡 RTX 4090，无显示器）

---

## 目录

1. [背景：Rerun 可视化框架](#1-背景rerun-可视化框架)
2. [三种 Headless 可视化方案对比](#2-三种-headless-可视化方案对比)
3. [方案 A：Web 浏览器直接访问（推荐）](#3-方案-a-web-浏览器直接访问推荐)
4. [方案 B：保存 .rrd 文件 + 本地 rerun 打开](#4-方案-b-保存-rrd-文件--本地-rerun-打开)
5. [方案 C：纯统计 + Matplotlib（完全 headless）](#5-方案-c-纯统计--matplotlib完全-headless)
6. [lerobot-dataset-viz 命令行工具](#6-lerobot-dataset-viz-命令行工具)
7. [数据集内容说明](#7-数据集内容说明)
8. [完整操作流程（Step by Step）](#8-完整操作流程step-by-step)
9. [常见问题](#9-常见问题)

---

## 1. 背景：Rerun 可视化框架

[Rerun](https://rerun.io) 是专为多模态时序数据设计的可视化工具，LeRobot 使用它展示：

- **相机流**：视频帧（`observation.images.*`）
- **机器人状态**：关节角度时序曲线（`observation.state`）
- **动作序列**：动作指令曲线（`action`）

```
┌─────────────────────────────────────────────────────────────┐
│                     Rerun 架构                               │
│                                                             │
│  LeRobot Dataset                                            │
│       │                                                     │
│       ▼                                                     │
│  lerobot-dataset-viz / viz_dataset_so101.py                 │
│       │  rr.log(image/state/action/...)                     │
│       ▼                                                     │
│  ┌─────────────┐    ┌──────────────────────────────────┐   │
│  │  gRPC 服务  │───►│  Web Viewer (浏览器 :9090)       │   │
│  │  (端口 9876)│    │  或桌面 rerun app                │   │
│  └─────────────┘    └──────────────────────────────────┘   │
│       │                                                     │
│       └──► 保存 .rrd 文件（离线查看）                        │
└─────────────────────────────────────────────────────────────┘
```

**安装**：
```bash
pip install rerun-sdk        # Rerun Python SDK
pip install lerobot          # 包含 lerobot-dataset-viz CLI 工具
```

---

## 2. 三种 Headless 可视化方案对比

| 方案 | 是否需要本地安装 | 实时性 | 操作复杂度 | 适用场景 |
|------|----------------|--------|----------|---------|
| **A. Web 浏览器** | 不需要（只需浏览器） | 实时 | 低 | **首选，最方便** |
| **B. 保存 .rrd** | 需要本地 rerun | 离线 | 中 | 需要本地精细查看 |
| **C. 纯统计** | 不需要 | 离线 | 低 | CI/CD、批量分析 |

---

## 3. 方案 A：Web 浏览器直接访问（推荐）

Rerun 内置 Web Viewer，在远端服务器启动后，本地浏览器直接打开即可，**无需在本地安装任何软件**。

### 3.1 远端启动（Docker 方式）

```bash
# 在远端节点 david@10.161.176.110 上执行
ssh david@10.161.176.110

# 方法一：使用封装好的脚本（推荐）
bash ~/github/lerobot_from_zero_to_expert/01_beginner/run_viz_docker.sh web 0

# 方法二：手动 docker run（注意 -p 暴露端口）
docker run --rm  -it --gpus all \
  -v ~/github/lerobot:/workspace/lerobot \
  -v ~/github/lerobot_from_zero_to_expert:/workspace/tutorial \
  -v ~/hf_cache:/root/.cache/huggingface \
  -v ~/lerobot_viz_output:/workspace/lerobot/outputs/viz \
  -w /workspace/lerobot \
  -p 9090:9090 \
  -p 9876:9876 \
  pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel \
  bash -c 'pip install -e "." -q \
           && pip uninstall torchcodec -y -q \
           && pip install rerun-sdk -q \
           && python /workspace/tutorial/01_beginner/viz_dataset_so101.py \
                --mode web \
                --episode-index 0 \
                --web-port 9090 \
                --grpc-port 9876'
```

启动成功后输出：
```
✓ Rerun Web Viewer 已启动
✓ 浏览器访问:  http://10.161.176.110:9090
✓ 桌面 rerun:  python -m rerun "rerun+http://10.161.176.110:9876/proxy"
```

### 3.2 本地访问

**直接访问**（如果服务器端口对外开放）：
```
浏览器打开: http://10.161.176.110:9090
```

**SSH 端口转发**（如果服务器有防火墙）：
```powershell
# 在本地 PowerShell 执行端口转发（保持终端开着，-N 表示只转发不执行命令）
ssh -L 9090:localhost:9090 -L 9876:localhost:9876 david@10.161.176.110 -N

# 方式一：浏览器打开（最简单，无需安装任何工具）
Start-Process "http://localhost:9090"

# 方式二：rerun 桌面客户端连接（Windows 下用 python -m rerun）
python -m rerun "rerun+http://localhost:9876/proxy"
```

### 3.3 使用 lerobot-dataset-viz CLI

不使用 Docker，直接在远端 conda 环境中运行：
```bash
# 远端（已安装 lerobot + rerun-sdk）
ssh david@10.161.176.110

# 激活环境（如果使用 conda）
conda activate lerobot

# 启动可视化服务
lerobot-dataset-viz \
  --repo-id lerobot/svla_so101_pickplace \
  --episode-index 0 \
  --mode distant \
  --grpc-port 9876 \
  --web-port 9090
```

### 3.4 Rerun Web Viewer 界面说明

浏览器打开后，界面分为三个区域：

```
┌──────────────────────────────────────────────────────────────────┐
│  Timeline (底部) ─ 拖动时间轴浏览帧                                │
├────────────────────────┬─────────────────────────────────────────┤
│  相机视图 (左上)        │  数据面板 (右)                           │
│  ┌──────────────────┐  │  ├── observation.images.up               │
│  │                  │  │  ├── observation.images.side              │
│  │  相机视频流       │  │  ├── action/0..5 (关节动作曲线)           │
│  │                  │  │  └── state/0..5 (关节状态曲线)            │
│  └──────────────────┘  │                                         │
└────────────────────────┴─────────────────────────────────────────┘
```

操作技巧：
- **播放/暂停**：空格键
- **逐帧**：左/右方向键
- **多 Panel**：拖拽数据项到不同区域分屏
- **导出截图**：右键菜单

---

## 4. 方案 B：保存 .rrd 文件 + 本地 rerun 打开

适合需要离线精细查看或分享给他人的场景。

### 4.1 远端保存 .rrd

```bash
# Docker 方式
bash ~/github/lerobot_from_zero_to_expert/01_beginner/run_viz_docker.sh save 0

# 或 lerobot-dataset-viz
lerobot-dataset-viz \
  --repo-id lerobot/svla_so101_pickplace \
  --episode-index 0 \
  --save 1 \
  --output-dir ~/lerobot_viz_output
```

生成文件：`~/lerobot_viz_output/lerobot_svla_so101_pickplace_episode_0.rrd`

### 4.2 拷贝到本地

```bash
# 在本地机器执行
scp david@10.161.176.110:~/lerobot_viz_output/lerobot_svla_so101_pickplace_episode_0.rrd .
```

### 4.3 本地打开

```powershell
# 安装 rerun（仅需一次）
pip install rerun-sdk

# Windows 下推荐用 python -m rerun（避免 PATH 问题）
python -m rerun lerobot_svla_so101_pickplace_episode_0.rrd
```

> **Windows 说明**：`pip install rerun-sdk` 后，`rerun.exe` 安装在 Python 的 Scripts 目录，但 Windows Store Python 不会自动将该目录加入 PATH，导致 PowerShell 找不到 `rerun` 命令。  

> **直接用 `python -m rerun`**（推荐，无需任何配置，立即可用）
>

---

## 5. 方案 C：纯统计 + Matplotlib（完全 headless）

无需 Rerun，适合 CI/CD 流水线、批量数据分析、或完全没有图形环境的节点。

### 5.1 运行

```bash
# Docker 方式
bash ~/github/lerobot_from_zero_to_expert/01_beginner/run_viz_docker.sh stats 0

# Python 直接运行（在 lerobot repo 根目录下，lerobot 已安装）
python ~/github/lerobot_from_zero_to_expert/01_beginner/viz_dataset_so101.py \
  --mode stats \
  --episode-index 0 1 2 \
  --output-dir outputs/viz
```

### 5.2 输出内容

脚本在 `outputs/viz/` 目录生成以下 PNG 文件：

| 文件 | 内容 |
|------|------|
| `action_episode_0.png` | 6维动作序列时间曲线 |
| `state_episode_0.png` | 6维关节状态时间曲线 |
| `frames_episode_0.png` | 相机截帧（每5秒一帧，双摄像头） |

### 5.3 拷贝到本地查看

```bash
# 在本地执行
scp -r david@10.161.176.110:~/lerobot_viz_output/ .
open lerobot_viz_output/frames_episode_0.png
```

---

## 6. lerobot-dataset-viz 命令行工具

lerobot 自带的 CLI 工具，支持与 `viz_dataset_so101.py` 类似的功能：

```bash
# 基础用法（本地可视化，需要本地 display）
lerobot-dataset-viz \
  --repo-id lerobot/svla_so101_pickplace \
  --episode-index 0

# 远端流式传输（headless 服务器）
# 远端执行:
lerobot-dataset-viz \
  --repo-id lerobot/svla_so101_pickplace \
  --episode-index 0 \
  --mode distant \
  --grpc-port 9876 \
  --web-port 9090

# 本地连接（Windows PowerShell）:
python -m rerun "rerun+http://10.161.176.110:9876/proxy"
# 或浏览器（无需安装任何工具）:
# http://10.161.176.110:9090

# 保存 .rrd（远端执行）:
lerobot-dataset-viz \
  --repo-id lerobot/svla_so101_pickplace \
  --episode-index 0 \
  --save 1 \
  --output-dir ~/viz_output

# 本地数据（不走 HuggingFace Hub）:
lerobot-dataset-viz \
  --repo-id lerobot/svla_so101_pickplace \
  --root /path/to/local/data \
  --mode local \
  --episode-index 0
```

---

## 7. 数据集内容说明

远端目录: ~/hf_cache/lerobot/lerobot/svla_so101_pickplace

`lerobot/svla_so101_pickplace` 是 SO-101 机器人抓取放置任务的真实演示数据集：

| 字段 | 说明 |
|------|------|
| `observation.images.up` | 俯视摄像头（H×W×3, RGB, 0~1 float） |
| `observation.images.side` | 侧视摄像头 |
| `observation.state` | 6维关节角度 (rad) |
| `action` | 6维关节目标角度 (rad) |
| `timestamp` | 帧时间戳 (秒) |
| `task` | 语言任务描述（如 "pick up the red block and place it in the box"） |

**数据统计**：
- 总帧数：11,939
- Episode 数：50
- FPS：30
- 单 Episode 时长：约 8 秒

---

## 8. 完整操作流程（Step by Step）

以下以 **方案 A（Web 浏览器）** 为例，演示完整操作流程。

### Step 1：确认远端服务器状态

```bash
# 检查 Docker 镜像是否存在
ssh david@10.161.176.110 "docker images pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel"

# 检查 GPU
ssh david@10.161.176.110 "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
```

### Step 2：启动可视化服务

```bash
# SSH 登录远端
ssh david@10.161.176.110

# 确保代码最新
cd ~/github/lerobot && git pull --ff-only

# 创建输出目录
mkdir -p ~/lerobot_viz_output

# 启动 Web 可视化（前台运行，Ctrl+C 停止）
bash ~/github/lerobot_from_zero_to_expert/01_beginner/run_viz_docker.sh web 0
```

### Step 3：SSH 端口转发（如有防火墙）

在**本地机器**新开一个终端：
```bash
# 端口转发（-N 表示只转发不执行命令，保持运行）
ssh -L 9090:localhost:9090 -L 9876:localhost:9876 \
    david@10.161.176.110 -N
```

### Step 4：本地浏览器访问

打开浏览器，访问：
```
http://localhost:9090
```

或者（如果端口对外开放）：
```
http://10.161.176.110:9090
```

### Step 5：浏览数据

在 Rerun Web Viewer 中：
1. 等待数据加载完毕（进度条消失）
2. 点击底部时间轴上的 ▶ 播放按钮
3. 在左侧面板展开 `observation.images.up` 和 `observation.images.side` 查看视频
4. 展开 `action/*` 查看关节动作曲线
5. 展开 `state/*` 查看关节状态曲线

---



## 参考资源

- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [Rerun 官网](https://rerun.io)
- [Rerun Python API](https://ref.rerun.io/docs/python/)
- [LeRobot 数据集在线查看器](https://huggingface.co/spaces/lerobot/visualize_dataset)
- [HuggingFace lerobot 文档](https://huggingface.co/docs/lerobot/en/using_dataset_tools)
