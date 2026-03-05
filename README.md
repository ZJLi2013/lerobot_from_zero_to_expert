# lerobot_from_zero_to_expert

> 从零到精通 LeRobot 机器人学习框架——面向 SO-101 机械臂的实战教程合集

本仓库是 [huggingface/lerobot](https://github.com/huggingface/lerobot) 的配套实践教程，所有脚本均已在两种 GPU 平台验证通过：

| 平台 | GPU | Docker 镜像 | 状态 |
|------|-----|------------|------|
| NVIDIA | RTX 4090 × 2 (24 GB) | `pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel` | ✅ 全部通过 |
| AMD ROCm | MI308X × 8 (192 GB HBM) | `rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0` | ✅ 全部通过 |

---

## 目录结构

```
lerobot_from_zero_to_expert/
├── 01_beginner/        # 入门：数据认知 + 三种策略训练 + 数据可视化
└── 02_intermediate/    # 进阶：合成数据生成（SDG）技术路径
```

---

## 01_beginner — 入门级

**目标**：在单张 4090 上跑通 ACT / Diffusion / SmolVLA 三种策略的训练 demo，并可视化真实机器人数据集。

| 文件 | 说明 |
|------|------|
| `test_diffusion_pusht.py` | Diffusion Policy 在 PushT 2D 仿真任务上的训练 demo |
| `test_act_so101.py` | ACT 在 SO-101 真实抓取数据上的训练 demo |
| `test_smolvla_so101.py` | SmolVLA（冻结 VLM 主干）在 SO-101 数据上的微调 demo |
| `viz_dataset_so101.py` | `svla_so101_pickplace` 数据集可视化（Web / Save / Stats 三种模式）|
| `run_smolvla_docker.sh` | SmolVLA Docker 一键启动脚本 |
| `run_viz_docker.sh` | 数据可视化 Docker 一键启动脚本 |
| `readme.md` | 入门指南：核心概念、数据格式、API 参考 |
| `reports.md` | 测试报告：各脚本实测结果、显存、Loss、耗时 |
| `data_visualize.md` | 可视化详细说明：headless 服务器方案、Rerun Web Viewer、SSH 转发 |

**快速开始**（远端节点）：

```bash
# 首次克隆
cd ~/github
git clone https://github.com/huggingface/lerobot.git
git clone https://github.com/ZJLi2013/lerobot_from_zero_to_expert.git

# 运行 SmolVLA 训练
bash ~/github/lerobot_from_zero_to_expert/01_beginner/run_smolvla_docker.sh

# 数据集可视化
bash ~/github/lerobot_from_zero_to_expert/01_beginner/run_viz_docker.sh stats 0
```

---

## 02_intermediate — 进阶级

**目标**：深入理解 LeRobot 数据结构，规划合成数据（SDG）管线，为构建大规模训练数据做准备。

| 文件 | 说明 |
|------|------|
| `sdg.md` | 合成数据生成技术指南：数据字段精解、四条 SDG 路径、Genesis 仿真评估 |

**核心内容**：

- **LeRobot 数据字段精解**：`observation.state` / `action` 的物理含义、单位（度）、与编码器的关系
- **SDG 四条技术路径**：
  - 路径 A：物理仿真采集（MuJoCo / Genesis / Isaac Sim）
  - 路径 B：视频运动重建（Video → IK → 关节角）
  - 路径 C：World Model 生成新视觉观测
  - 路径 D：数据增强（最快上手）
- **Genesis 可行性评估**：Genesis 与 SO-101 / LeRobot 接口天然兼容，10-80x 加速，推荐优先级最高

---

## 环境说明

所有脚本通过 Docker 挂载双仓库运行，无需在本地安装 lerobot：

```bash
docker run --rm --gpus all \
  -v ~/github/lerobot:/workspace/lerobot \                        # pip install 用
  -v ~/github/lerobot_from_zero_to_expert:/workspace/tutorial \   # 脚本文件
  -v ~/hf_cache:/root/.cache/huggingface \                        # 数据集缓存
  -w /workspace/lerobot \
  pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel \
  bash -c 'pip install -e ".[smolvla]" -q \
           && pip uninstall torchcodec -y -q \
           && python /workspace/tutorial/01_beginner/test_smolvla_so101.py'
```

> **注意**：PyTorch 2.8.0 与 torchcodec 0.10.0 存在 ABI 不兼容，`pip uninstall torchcodec` 是必要步骤。

---

## 路线图

```
✅ 01_beginner    — ACT / Diffusion / SmolVLA 训练 demo + 数据可视化
✅ 02_intermediate — SDG 技术方案 + Genesis 评估

🔲 03_advanced    — Genesis 仿真采集管线（SO-101 MuJoCo + 并行环境）
🔲 04_expert      — SmolVLA / Pi0 大规模微调 + sim-to-real 验证
```

---

## 参考

- [huggingface/lerobot](https://github.com/huggingface/lerobot)
- [SO-ARM100 机械臂](https://github.com/TheRobotStudio/SO-ARM100)
- [Genesis 物理仿真引擎](https://github.com/Genesis-Embodied-AI/Genesis)
- [robosim-so101-pickup-v2（MuJoCo 仿真 SO-101 数据集）](https://huggingface.co/datasets/houmans/robosim-so101-pickup-v2)
