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
├── 01_beginner/           # 入门：数据认知 + 三种策略训练 + 数据可视化
└── 02_intermediate/       # 进阶：Genesis × SO-101 合成数据采集管线
    ├── doc/               #   设计文档 + 调参手册
    └── scripts/           #   可运行脚本（POC → 基础采集 → 调参 → 并行生产）
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

**目标**：在 Genesis 物理仿真中构建 SO-101 合成数据采集管线（SDG），从单环境验证到并行批量生产，输出可训练的 LeRobot v3 数据集。

| 文件 / 目录 | 说明 |
|-------------|------|
| `doc/implement_guide.md` | 实现指南：架构设计、环境准备、脚本说明、运行流程、最小命令清单 |
| `doc/best_practices.md` | 调参与验收手册：基线配置、offset/gripper 调参法、判定指标 |
| `doc/survey.md` | SDG 技术调研：四条路径对比、Genesis 可行性评估 |
| `scripts/1_poc_pipeline.py` | Genesis POC 验证管线 |
| `scripts/2_collect.py` | SO-101 采集（probe + 朝向修正 + .rrd 输出） |
| `scripts/3_grasp_experiment.py` | 抓取调参实验（推荐入口），自动 offset 搜索 + metrics.json |
| `scripts/4_parallel_lerobot.py` | 并行批量采集（N_ENVS 并行 + 域随机化 + 直接写 LeRobot 格式） |
| `scripts/npy_to_lerobot.py` | npy → LeRobot v3 格式转换 |

**快速开始**（远端 4090 节点）：

```bash
ssh david@<4090_HOST> "mkdir -p ~/sdg_grasp_exp && docker run --rm --gpus all \
  -v ~/github/lerobot_from_zero_to_expert:/workspace/lfzte \
  -v ~/sdg_grasp_exp:/output \
  genesis_poc:latest \
  python -u /workspace/lfzte/02_intermediate/scripts/3_grasp_experiment.py \
  --exp-id E3_auto_offset --episodes 1 --episode-length 6 --save /output \
  --auto-tune-offset --gripper-open 70 --gripper-close 20 --close-hold-steps 12"
```

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
✅ 01_beginner     — ACT / Diffusion / SmolVLA 训练 demo + 数据可视化
✅ 02_intermediate  — Genesis × SO-101 合成数据采集管线（POC → 调参 → 并行生产）

🔲 03_advanced     — 大规模 SDG 数据集 + 域随机化 + SmolVLA / Pi0 微调
🔲 04_expert       — sim-to-real 迁移验证 + 真机部署
```

---

## 参考

- [huggingface/lerobot](https://github.com/huggingface/lerobot)
- [SO-ARM100 机械臂](https://github.com/TheRobotStudio/SO-ARM100)
- [Genesis 物理仿真引擎](https://github.com/Genesis-Embodied-AI/Genesis)
- [robosim-so101-pickup-v2（MuJoCo 仿真 SO-101 数据集）](https://huggingface.co/datasets/houmans/robosim-so101-pickup-v2)
