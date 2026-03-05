# LeRobot 入门指南

LeRobot 是 Hugging Face 开发的机器人学习框架，提供统一的数据集格式、预训练模型和训练工具。

## 目录

1. [本目录文件说明](#本目录文件说明)
2. [核心概念](#核心概念)
3. [LeRobot 数据格式 (v3.0)](#lerobot-数据格式-v30)
4. [训练 Pipeline 概览](#训练-pipeline-概览)
5. [关键概念深入](#关键概念深入)
6. [常见问题](#常见问题)
7. [参考资源](#参考资源)

---

## 本目录文件说明

> 所有脚本均在远端节点（`david@<4090_HOST>`，RTX 4090）通过 Docker 测试通过。  
> 详细测试结果见 [reports.md](reports.md)，可视化说明见 [data_visualize.md](data_visualize.md)。

### 可执行脚本

| 文件 | 任务 | 策略 | 数据集 | 显存 |
|------|------|------|--------|------|
| `test_diffusion_pusht.py` | PushT 推圆盘（2D仿真） | Diffusion Policy | `lerobot/pusht` | ~8GB |
| `test_act_so101.py` | SO-101 pick-place（真实机器人数据） | ACT | `lerobot/svla_so101_pickplace` | ~6GB |
| `test_smolvla_so101.py` | SO-101 pick-place（VLA微调） | SmolVLA | `lerobot/svla_so101_pickplace` | ~14GB |
| `viz_dataset_so101.py` | 数据集可视化（Web/Save/Stats三种模式） | — | `lerobot/svla_so101_pickplace` | — |

### Docker 启动脚本

| 文件 | 说明 |
|------|------|
| `run_smolvla_docker.sh` | 一键运行 SmolVLA 训练（自动安装依赖） |
| `run_viz_docker.sh` | 一键运行数据集可视化（支持 web/save/stats 三种模式） |

### 运行方式

> ✅ 已在 **NVIDIA RTX 4090**（CUDA 12.6）和 **AMD Instinct MI308X**（ROCm 6.4.3）双平台验证通过，详见 [reports.md](reports.md)。

远端节点首次准备：
```bash
cd ~/github
git clone https://github.com/huggingface/lerobot.git
git clone https://github.com/ZJLi2013/lerobot_from_zero_to_expert.git
mkdir -p ~/hf_cache
```

#### NVIDIA 平台（`--gpus all`）

```bash
# Diffusion Policy + PushT
docker run --rm --gpus all \
  -v ~/github/lerobot:/workspace/lerobot \
  -v ~/github/lerobot_from_zero_to_expert:/workspace/tutorial \
  -v ~/hf_cache:/root/.cache/huggingface \
  -w /workspace/lerobot \
  pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel \
  bash -c 'pip install -e ".[pusht]" -q && pip uninstall torchcodec -y -q \
           && python /workspace/tutorial/01_beginner/test_diffusion_pusht.py'

# ACT + SO-101
docker run --rm --gpus all \
  -v ~/github/lerobot:/workspace/lerobot \
  -v ~/github/lerobot_from_zero_to_expert:/workspace/tutorial \
  -v ~/hf_cache:/root/.cache/huggingface \
  -w /workspace/lerobot \
  pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel \
  bash -c 'pip install -e . -q && pip uninstall torchcodec -y -q \
           && python /workspace/tutorial/01_beginner/test_act_so101.py'

# SmolVLA + SO-101（或用 bash 01_beginner/run_smolvla_docker.sh）
docker run --rm --gpus all \
  -v ~/github/lerobot:/workspace/lerobot \
  -v ~/github/lerobot_from_zero_to_expert:/workspace/tutorial \
  -v ~/hf_cache:/root/.cache/huggingface \
  -w /workspace/lerobot \
  pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel \
  bash -c 'pip install -e ".[smolvla]" -q && pip uninstall torchcodec -y -q \
           && python /workspace/tutorial/01_beginner/test_smolvla_so101.py'
```

#### AMD ROCm 平台（`--device /dev/kfd --device /dev/dri --group-add video`）



```bash
AMD_FLAGS="--device /dev/kfd --device /dev/dri --group-add video"
AMD_IMG="rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0"

# Diffusion Policy + PushT
docker run --rm $AMD_FLAGS \
  -v ~/github/lerobot:/workspace/lerobot \
  -v ~/github/lerobot_from_zero_to_expert:/workspace/tutorial \
  -v ~/hf_cache:/root/.cache/huggingface \
  -w /workspace/lerobot \
  $AMD_IMG \
  bash -c 'pip install -e ".[pusht]" -q && pip uninstall torchcodec -y -q \
           && python3 /workspace/tutorial/01_beginner/test_diffusion_pusht.py'

# ACT / SmolVLA：将 ".[pusht]" 分别换为 "." 和 ".[smolvla]"，脚本路径相应替换
```

> **ROCm 说明**：PyTorch ROCm 版本将 `torch.cuda.*` 映射到 HIP，Python 脚本无需修改。  
> 两平台均需 `pip uninstall torchcodec`（libavdevice.so 缺失），回退到 `pyav` 解码视频。

**数据集可视化**（见 [data_visualize.md](data_visualize.md) 的完整说明）：
```bash
bash 01_beginner/run_viz_docker.sh stats 0   # 统计+Matplotlib图表
bash 01_beginner/run_viz_docker.sh web   0   # Rerun Web Viewer（浏览器访问）
bash 01_beginner/run_viz_docker.sh save  0   # 保存 .rrd 文件
```

---

## 核心概念

### 1. 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                        LeRobot Framework                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │   Robots    │    │  Datasets   │    │     Policies        │ │
│  │  (硬件控制)  │◄──►│  (数据格式)  │◄──►│   (策略模型)         │ │
│  └─────────────┘    └─────────────┘    └─────────────────────┘ │
│         │                  │                      │             │
│         ▼                  ▼                      ▼             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │ Teleop      │    │ Transforms  │    │  Pre/Post           │ │
│  │ (遥控操作)   │    │ (数据增强)   │    │  Processors         │ │
│  └─────────────┘    └─────────────┘    └─────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 核心组件

| 组件 | 说明 |
|------|------|
| **LeRobotDataset** | 标准化的数据集格式，支持多模态数据（视觉+状态+动作） |
| **PreTrainedPolicy** | 策略模型基类，支持 ACT、Diffusion、VLA 等多种架构 |
| **Preprocessor/Postprocessor** | 数据预处理和后处理管道 |
| **Robot** | 硬件抽象接口，支持多种机器人 |

### 3. 支持的策略类型

| 类别 | 模型 | 说明 |
|------|------|------|
| **模仿学习 (IL)** | ACT, Diffusion, VQ-BeT | 从演示数据学习 |
| **强化学习 (RL)** | HIL-SERL, TDMPC, SAC | 通过奖励信号学习 |
| **视觉语言动作 (VLA)** | Pi0, SmolVLA, GR00T | 多模态输入的端到端模型 |

### 4. 策略横向对比

| 特性 | ACT | Diffusion Policy | SmolVLA |
|------|-----|------------------|---------|
| **架构** | Transformer Enc-Dec | U-Net + 扩散 | VLM + Action Expert |
| **动作预测** | 直接回归 (Chunking) | 迭代去噪 | Flow Matching |
| **推理速度** | 快 | 慢 | 中 |
| **语言条件** | ✗ | ✗ | ✓ |
| **显存 (SO-101)** | ~6GB | ~8GB | ~14GB |
| **模型参数量** | ~45M | ~55M | ~450M |
| **预训练来源** | 无 | 无 | HF smolvla_base |

---

## LeRobot 数据格式 (v3.0)

### 1. 目录结构

```
dataset_root/
├── meta/
│   ├── info.json              # 数据集元信息（特征、形状、FPS等）
│   ├── stats.json             # 统计信息（均值、标准差，用于归一化）
│   ├── tasks.parquet          # 任务描述列表
│   └── episodes/              # Episode 元数据
├── data/
│   └── chunk-000/
│       └── file-000.parquet   # 表格数据（状态、动作、时间戳）
└── videos/
    └── observation.images.up/
        └── chunk-000/
            └── episode_000000.mp4
```

### 2. 核心字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `observation.state` | float32 (6,) | 当前关节位置（度）|
| `action` | float32 (6,) | 目标关节位置（度）|
| `observation.images.*` | video (H,W,3) | 摄像头图像，存为 MP4 |
| `timestamp` | float32 | 帧时间戳（秒）|
| `frame_index` | int64 | episode 内帧序号 |
| `episode_index` | int64 | episode 编号 |
| `task` | str | 自然语言任务描述（每帧都有）|

### 3. 数据访问 API

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

# 仅加载元数据（不下载实际数据）
meta = LeRobotDatasetMetadata("lerobot/svla_so101_pickplace")
print(f"Episodes: {meta.total_episodes}, FPS: {meta.fps}")
print(f"Features: {list(meta.features.keys())}")

# 加载数据集（支持只加载部分 episodes）
dataset = LeRobotDataset("lerobot/svla_so101_pickplace", episodes=[0, 1])

# 时间窗口访问（action chunking）
delta_timestamps = {
    "observation.state": [0],
    "action": [t / dataset.fps for t in range(50)],   # 未来 50 帧动作
    "observation.images.up": [0],
}
dataset = LeRobotDataset("lerobot/svla_so101_pickplace", delta_timestamps=delta_timestamps)

sample = dataset[0]
print(sample["observation.state"].shape)   # torch.Size([6])
print(sample["action"].shape)              # torch.Size([50, 6])
print(sample["observation.images.up"].shape)  # torch.Size([3, H, W])
```

---

## 训练 Pipeline 概览

```
┌──────────┐    ┌─────────────┐    ┌──────────┐    ┌─────────┐
│ Dataset  │───►│Preprocessor │───►│  Policy  │───►│  Loss   │
│ (数据集)  │    │   (预处理)   │    │  (策略)  │    │ (损失)   │
└──────────┘    └─────────────┘    └──────────┘    └────┬────┘
                                                        │
                ┌─────────────┐    ┌──────────┐         │
                │  Optimizer  │◄───│ Backward │◄────────┘
                │  (优化器)    │    │ (反向传播)│
                └──────┬──────┘    └──────────┘
                       │
                ┌──────▼──────┐
                │ Checkpoint  │
                │ (保存/上传)  │
                └─────────────┘
```

### 核心训练代码模式

> 完整可运行脚本见同目录 `test_*.py` 文件。

```python
# 1. 加载数据集元数据
meta = LeRobotDatasetMetadata(dataset_id)
features = dataset_to_policy_features(meta.features)
output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
input_features  = {k: v for k, v in features.items() if k not in output_features}

# 2. 创建策略
cfg = ACTConfig(input_features=input_features, output_features=output_features)
policy = ACTPolicy(cfg).to(device)

# 3. 预处理/后处理器
preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=meta.stats)

# 4. 数据加载器
dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# 5. 训练循环
optimizer = cfg.get_optimizer_preset().build(policy.parameters())
for batch in dataloader:
    batch = preprocessor(batch)
    loss, _ = policy.forward(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# 6. 保存
policy.save_pretrained(output_dir)
preprocessor.save_pretrained(output_dir)
postprocessor.save_pretrained(output_dir)
```

### 命令行方式（lerobot-train CLI）

```bash
# Diffusion Policy
lerobot-train --policy=diffusion --dataset.repo_id=lerobot/pusht \
  --output_dir=outputs/diffusion_pusht --batch_size=64 --steps=5000

# ACT
lerobot-train --policy=act --dataset.repo_id=lerobot/svla_so101_pickplace \
  --output_dir=outputs/act_so101 --batch_size=32 --steps=10000

# 从检查点恢复
lerobot-train --config_path=outputs/act_so101/checkpoints/last/train_config.json --resume=true
```

### 创建自定义数据集

```python
dataset = LeRobotDataset.create(
    repo_id="your_name/my_dataset",
    fps=30,
    features={
        "observation.state": {"dtype": "float32", "shape": (6,), "names": [...]},
        "action":            {"dtype": "float32", "shape": (6,), "names": [...]},
        "observation.images.front": {"dtype": "video", "shape": (480, 640, 3), "names": None},
    },
)

for episode_data in your_data:
    for frame in episode_data:
        dataset.add_frame({
            "observation.state": frame["state"],
            "action":            frame["action"],
            "observation.images.front": frame["image"],   # uint8 (H,W,3)
            "task": "pick up the cube",
        })
    dataset.save_episode(task="pick up the cube")

dataset.consolidate(run_compute_stats=True)
# dataset.push_to_hub()
```

### 多数据集训练

```python
from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset

dataset = MultiLeRobotDataset([
    "lerobot/svla_so101_pickplace",
    "your_name/so101_synthetic",   # 真实 + 合成数据联合训练
])
```

---

## 关键概念深入

### 1. Action Chunking

ACT 模型的核心：一次预测多步动作序列，而非单步：

```python
# 普通策略：每步预测一个动作
action_t = policy(obs_t)           # shape: (6,)

# Action Chunking：预测未来 chunk_size 步
action_chunk = policy(obs_t)       # shape: (100, 6)

# 时间窗口配置
delta_timestamps = {
    "action": [t / fps for t in range(chunk_size)],   # 未来 100 帧
}
```

### 2. 归一化

LeRobot 使用 `stats.json` 中的统计信息做自动归一化：

```python
# stats.json 存储 min/max/mean/std
# make_pre_post_processors 自动处理：
#   - state/action: mean_std 或 min_max 归一化
#   - images: 归一化到 [0,1]（float32）
preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=meta.stats)
```

### 3. 视觉编码器对比

| 策略 | 视觉编码器 | 特点 |
|------|-----------|------|
| ACT | ResNet18/34 | 轻量，适合精细操作 |
| Diffusion | ResNet + FiLM | 条件融合 |
| SmolVLA | SigLIP | 预训练视觉语言对齐 |
| Pi0 | PaliGemma | 大规模预训练 |
| GR00T | Eagle | NVIDIA 通用机器人 |

### 4. SmolVLA 特殊设计

SmolVLA 冻结视觉语言主干，只微调 Action Expert：

```python
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

cfg = SmolVLAConfig(
    input_features=input_features,
    output_features=output_features,
    freeze_vision_encoder=True,   # 冻结 SigLIP 视觉编码器
    train_expert_only=True,        # 只训练 action expert 部分
)
# 约 450M 参数，实际训练参数量远少于全量微调
```

---

## 常见问题

### Q1: 如何选择合适的策略？

| 场景 | 推荐策略 | 原因 |
|------|---------|------|
| 简单任务，少量数据 | ACT | 训练快，稳定，~45M参数 |
| 复杂轨迹，强多模态 | Diffusion | 表达能力强 |
| 语言条件任务 | SmolVLA / Pi0 | 支持自然语言指令 |
| 大规模预训练 | GR00T | 强大泛化能力 |

### Q2: torchcodec 兼容性问题

PyTorch 2.8.0 与 torchcodec 0.10.0 存在 ABI 不兼容，需在安装 lerobot 后立即卸载：

```bash
pip install -e ".[smolvla]" && pip uninstall torchcodec -y
```

lerobot 会自动回退到 `pyav` 解码视频，功能完全正常。

### Q3: 训练不稳定

```python
# 1. 降低学习率（从 1e-4 → 1e-5）
# 2. 增加梯度裁剪
torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm=1.0)
# 3. 确保 dataset_stats 不为 None（用于归一化）
assert meta.stats is not None
```

### Q4: 如何处理不同相机名称

```python
# 用 rename_map 映射相机键名（数据集列名 → 策略期望名）
policy = make_policy(cfg=policy_config, ds_meta=meta, rename_map={
    "observation.images.left_cam": "observation.images.camera1",
})
```

---

## 参考资源

### 官方文档
- [LeRobot 文档](https://huggingface.co/docs/lerobot/index)
- [LeRobotDataset v3.0 格式说明](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- [Hardware 集成指南](https://huggingface.co/docs/lerobot/integrate_hardware)

### 中文教程
- [LeRobot+SO-ARM101 中文教程 - 同济子豪兄](https://zihao-ai.feishu.cn/wiki/space/7589642043471924447)
- [LeRobot Tutorial CN](https://github.com/CSCSX/LeRobotTutorial-CN)

### 论文
- [ACT: Learning Fine-Grained Bimanual Manipulation](https://arxiv.org/abs/2304.13705)
- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)
- [SmolVLA: An efficient VLA for affordable and fast robot learning](https://huggingface.co/blog/smolvla)

### 数据集
- [Hugging Face LeRobot 数据集](https://huggingface.co/lerobot)
- [社区数据集浏览](https://huggingface.co/datasets?other=LeRobot)

### 社区
- [Discord](https://discord.gg/q8Dzzpym3f)
- [GitHub lerobot](https://github.com/huggingface/lerobot)
