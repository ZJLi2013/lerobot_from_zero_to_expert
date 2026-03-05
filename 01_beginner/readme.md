# LeRobot 学习指南

LeRobot 是 Hugging Face 开发的机器人学习框架，提供了统一的数据集格式、预训练模型和训练工具，使机器人学习变得更加简单。

## 目录

1. [核心概念](#核心概念)
2. [LeRobot 数据格式 (v3.0)](#lerobot-数据格式-v30)
3. [Post-Training Pipeline](#post-training-pipeline)
4. [由浅入深的学习样例](#由浅入深的学习样例)
5. [参考资源](#参考资源)

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
│       └── chunk-000/
│           └── file-000.parquet
├── data/
│   └── chunk-000/
│       ├── file-000.parquet   # 表格数据（状态、动作、时间戳）
│       └── file-001.parquet
└── videos/
    └── observation.images.front/
        └── chunk-000/
            ├── file-000.mp4   # 视频数据
            └── file-001.mp4
```

### 2. 核心特征 (Features)

```python
# info.json 中的 features 定义示例
{
    "features": {
        "observation.state": {
            "dtype": "float32",
            "shape": [6],          # 6维状态向量
            "names": ["joint_1", "joint_2", ...],
        },
        "action": {
            "dtype": "float32",
            "shape": [6],          # 6维动作向量
            "names": ["joint_1", "joint_2", ...],
        },
        "observation.images.front": {
            "dtype": "video",      # 视频类型
            "shape": [480, 640, 3], # (H, W, C)
            "info": {
                "video.fps": 30,
                "video.codec": "libsvtav1"
            }
        }
    }
}
```

### 3. 数据访问模式

#### 3.1 基本访问

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 加载数据集
dataset = LeRobotDataset("lerobot/aloha_mobile_cabinet")

# 访问单个样本
sample = dataset[0]
print(sample.keys())
# dict_keys(['observation.state', 'action', 'observation.images.front', 
#            'timestamp', 'episode_index', 'frame_index', 'task_index'])

# 获取形状
print(sample['observation.state'].shape)  # torch.Size([6])
print(sample['action'].shape)             # torch.Size([6])
print(sample['observation.images.front'].shape)  # torch.Size([3, 480, 640])
```

#### 3.2 时间窗口访问 (Delta Timestamps)

```python
# 定义时间偏移
delta_timestamps = {
    # 加载当前帧及前 3 帧（历史观测）
    "observation.images.front": [-0.3, -0.2, -0.1, 0],
    "observation.state": [-0.2, -0.1, 0],
    # 加载未来 16 帧的动作（动作预测）
    "action": [t / dataset.fps for t in range(16)],
}

dataset = LeRobotDataset(
    "lerobot/aloha_mobile_cabinet",
    delta_timestamps=delta_timestamps
)

sample = dataset[100]
print(sample['observation.images.front'].shape)  # torch.Size([4, 3, 480, 640])
print(sample['observation.state'].shape)         # torch.Size([3, 6])
print(sample['action'].shape)                    # torch.Size([16, 6])
```

### 4. 元数据结构

```python
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

# 仅加载元数据（不下载实际数据）
meta = LeRobotDatasetMetadata("lerobot/aloha_mobile_cabinet")

print(f"总 Episode 数: {meta.total_episodes}")
print(f"总帧数: {meta.total_frames}")
print(f"FPS: {meta.fps}")
print(f"机器人类型: {meta.robot_type}")
print(f"相机键名: {meta.camera_keys}")
print(f"特征: {meta.features}")
print(f"统计信息: {meta.stats}")  # 用于归一化
```

---

## Post-Training Pipeline

### 快速入门：单卡训练 Demo (RTX 4090)

这是最轻量级的训练入门示例，适合在单张 4090 (24GB) 上快速验证训练流程。

#### 环境配置

```bash
# 1. 创建 conda 环境
conda create -n lerobot python=3.10
conda activate lerobot

# 2. 安装 PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. 安装 LeRobot + PushT 仿真环境依赖
pip install -e ".[pusht]"
```

#### 最简训练脚本 (Diffusion Policy + PushT)

```python
"""
单卡训练 Demo: Diffusion Policy 在 PushT 环境上的训练
- 数据集: lerobot/pusht (2D 推箱子任务，数据量小)
- 策略: Diffusion Policy
- 显存: ~8-10GB (batch_size=64)
- 训练时间: ~10-15分钟 (5000步)
"""
from pathlib import Path
import torch
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors


def main():
    # ===== 配置 =====
    output_directory = Path("outputs/train/pusht_diffusion_demo")
    output_directory.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda")
    training_steps = 5000  # 5000步即可获得可评估的结果
    log_freq = 100
    batch_size = 64  # 4090可以使用较大batch size
    
    # ===== 1. 加载数据集元数据 =====
    print("1. 加载数据集元数据...")
    dataset_metadata = LeRobotDatasetMetadata("lerobot/pusht")
    
    # 将数据集特征转换为策略所需的特征格式
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    # ===== 2. 创建策略 =====
    print("2. 创建 Diffusion Policy...")
    cfg = DiffusionConfig(input_features=input_features, output_features=output_features)
    policy = DiffusionPolicy(cfg)
    policy.train()
    policy.to(device)
    
    # 创建预处理/后处理器
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)
    
    # ===== 3. 配置时间窗口 =====
    # Diffusion Policy 需要历史观测和未来动作序列
    delta_timestamps = {
        "observation.image": [-0.1, 0.0],  # 当前帧 + 前一帧
        "observation.state": [-0.1, 0.0],
        "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    }
    
    # ===== 4. 创建数据加载器 =====
    print("3. 准备数据加载器...")
    dataset = LeRobotDataset("lerobot/pusht", delta_timestamps=delta_timestamps)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    
    # ===== 5. 设置优化器 =====
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    
    # ===== 6. 训练循环 =====
    print("4. 开始训练...")
    print(f"   - 设备: {device}")
    print(f"   - Batch Size: {batch_size}")
    print(f"   - 训练步数: {training_steps}")
    
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"   Step: {step:5d} | Loss: {loss.item():.4f}")
            
            step += 1
            if step >= training_steps:
                done = True
                break

    # ===== 7. 保存模型 =====
    print("5. 保存模型...")
    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)
    print(f"   模型已保存到: {output_directory}")


if __name__ == "__main__":
    main()
```

#### 显存占用参考

| Batch Size | 预估显存 | 适用GPU |
|------------|---------|---------|
| 64 | ~8-10GB | RTX 4090 / A100 |
| 32 | ~5-6GB | RTX 3090 / 4080 |
| 16 | ~3-4GB | RTX 3080 / 4070 |

#### 命令行方式训练

```bash
# 使用 CLI 工具进行训练
python examples/training/train_policy.py

# 或使用 lerobot-train 命令
lerobot-train \
  --policy=diffusion \
  --dataset.repo_id=lerobot/pusht \
  --output_dir=outputs/diffusion_pusht \
  --batch_size=64 \
  --steps=5000
```

---

### ACT + 真实机器人数据训练 Demo

这是更接近实际机器人应用的训练场景，使用真实采集的机械臂数据进行 ACT (Action Chunking Transformer) 训练。

#### 环境配置

```bash
# 使用上面相同的环境
conda activate lerobot

# ACT 不需要额外依赖，基础安装即可
pip install -e .
```

#### ACT 训练脚本

```python
"""
ACT 训练 Demo: 使用真实机器人数据训练 ACT 策略
- 数据集: lerobot/svla_so101_pickplace (真实机器人抓取数据)
- 策略: ACT (Action Chunking Transformer)
- 显存: ~6-8GB (batch_size=32)
- 特点: 更接近实际机器人部署场景
"""
from pathlib import Path
import torch
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    """将帧索引转换为时间戳"""
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]


def main():
    # ===== 配置 =====
    output_directory = Path("outputs/train/act_so101_demo")
    output_directory.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda")
    dataset_id = "lerobot/svla_so101_pickplace"  # 真实机器人抓取数据
    training_steps = 1000  # demo 目的，实际训练需要更多步数
    log_freq = 50
    batch_size = 32  # ACT 通常使用较小的 batch size
    
    # ===== 1. 加载数据集元数据 =====
    print("1. 加载数据集元数据...")
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    print(f"   - 数据集: {dataset_id}")
    print(f"   - 总帧数: {dataset_metadata.total_frames}")
    print(f"   - FPS: {dataset_metadata.fps}")
    
    # 将数据集特征转换为策略所需的特征格式
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    # ===== 2. 创建 ACT 策略 =====
    print("2. 创建 ACT 策略...")
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        # ACT 核心参数
        chunk_size=100,       # 动作预测序列长度
        n_obs_steps=1,        # 观测历史长度
        dim_model=256,        # Transformer 隐藏层维度
        n_heads=8,            # 注意力头数
        n_encoder_layers=4,   # 编码器层数
        n_decoder_layers=1,   # 解码器层数
    )
    
    policy = ACTPolicy(cfg)
    policy.train()
    policy.to(device)
    
    print(f"   - 模型参数量: {sum(p.numel() for p in policy.parameters()):,}")
    
    # 创建预处理/后处理器
    preprocessor, postprocessor = make_pre_post_processors(
        cfg, 
        dataset_stats=dataset_metadata.stats
    )
    
    # ===== 3. 配置时间窗口 (Action Chunking) =====
    # ACT 核心: 预测未来 chunk_size 个动作
    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }
    
    # 添加图像特征的时间窗口
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
        for k in cfg.image_features
    }
    
    # ===== 4. 创建数据加载器 =====
    print("3. 准备数据加载器...")
    dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    
    # ===== 5. 设置优化器 =====
    # ACT 使用其内置的优化器配置
    optimizer = cfg.get_optimizer_preset().build(policy.parameters())
    
    # ===== 6. 训练循环 =====
    print("4. 开始训练...")
    print(f"   - 设备: {device}")
    print(f"   - Batch Size: {batch_size}")
    print(f"   - 训练步数: {training_steps}")
    print(f"   - Chunk Size: {cfg.chunk_size}")
    
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, info = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"   Step: {step:5d} | Loss: {loss.item():.4f}")
            
            step += 1
            if step >= training_steps:
                done = True
                break

    # ===== 7. 保存模型 =====
    print("5. 保存模型...")
    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)
    print(f"   模型已保存到: {output_directory}")
    
    # 可选: 上传到 Hugging Face Hub
    # policy.push_to_hub("your-username/act_so101_demo")


if __name__ == "__main__":
    main()
```

#### ACT vs Diffusion 对比

| 特性 | ACT | Diffusion Policy |
|------|-----|------------------|
| **架构** | Transformer Encoder-Decoder | U-Net + 扩散过程 |
| **动作预测** | 直接回归 (Action Chunking) | 迭代去噪 |
| **推理速度** | 快 (单次前向传播) | 慢 (多步去噪) |
| **显存占用** | 较低 | 较高 |
| **适用场景** | 精细操作、双臂协作 | 复杂多模态任务 |
| **训练稳定性** | 稳定 | 需要调参 |

#### 命令行方式训练 ACT

```bash
# 使用 lerobot-train CLI
lerobot-train \
  --policy=act \
  --dataset.repo_id=lerobot/svla_so101_pickplace \
  --output_dir=outputs/act_so101 \
  --batch_size=32 \
  --steps=10000 \
  --save_freq=2000 \
  --log_freq=100

# 或运行教程脚本
python examples/tutorial/act/act_training_example.py
```

#### 推荐的训练数据集

| 数据集 | 说明 | 难度 |
|--------|------|------|
| `lerobot/svla_so101_pickplace` | SO-101 机械臂抓取 | 入门 |
| `lerobot/aloha_sim_insertion_human` | ALOHA 仿真插入任务 | 中级 |
| `lerobot/aloha_mobile_cabinet` | ALOHA 移动机器人开柜 | 高级 |

---

### 1. 训练流程概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Pipeline                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌─────────────┐    ┌──────────┐    ┌─────────┐ │
│  │ Dataset  │───►│Preprocessor │───►│  Policy  │───►│  Loss   │ │
│  │ (数据集)  │    │   (预处理)   │    │  (策略)  │    │ (损失)   │ │
│  └──────────┘    └─────────────┘    └──────────┘    └────┬────┘ │
│                                                          │      │
│                    ┌─────────────┐    ┌──────────┐       │      │
│                    │  Optimizer  │◄───│ Backward │◄──────┘      │
│                    │  (优化器)    │    │ (反向传播) │             │
│                    └─────────────┘    └──────────┘              │
│                           │                                     │
│                           ▼                                     │
│                    ┌─────────────┐    ┌──────────┐              │
│                    │ Checkpoint  │───►│  Hub     │              │
│                    │ (保存检查点) │    │ (上传)    │              │
│                    └─────────────┘    └──────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 配置系统

LeRobot 使用 `TrainPipelineConfig` 统一管理训练配置：

```python
from dataclasses import dataclass, field
from lerobot.configs.train import TrainPipelineConfig

@dataclass
class TrainPipelineConfig:
    # 数据集配置
    dataset: DatasetConfig
    
    # 策略配置
    policy: PreTrainedConfig | None = None
    
    # 环境配置（可选，用于在线评估）
    env: EnvConfig | None = None
    
    # 训练参数
    output_dir: Path | None = None
    seed: int | None = 1000
    num_workers: int = 4
    batch_size: int = 8
    steps: int = 100_000
    
    # 日志和检查点
    eval_freq: int = 20_000
    log_freq: int = 200
    save_freq: int = 20_000
    
    # 优化器配置
    optimizer: OptimizerConfig | None = None
    scheduler: LRSchedulerConfig | None = None
```

### 3. 预处理/后处理管道

```python
# 预处理管道示意
Preprocessor Pipeline:
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ DeviceMove   │───►│ Normalize    │───►│ ImageResize  │
│ (移到GPU)     │    │ (归一化)      │    │ (图像缩放)    │
└──────────────┘    └──────────────┘    └──────────────┘

# 后处理管道示意
Postprocessor Pipeline:
┌──────────────┐    ┌──────────────┐
│ Unnormalize  │───►│ ActionClip   │
│ (反归一化)    │    │ (动作裁剪)    │
└──────────────┘    └──────────────┘
```

### 4. 训练循环核心代码

```python
def update_policy(
    policy: PreTrainedPolicy,
    batch: dict,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
) -> tuple[dict, dict]:
    """单步训练更新"""
    policy.train()
    
    # 前向传播
    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)
    
    # 反向传播
    accelerator.backward(loss)
    
    # 梯度裁剪
    if grad_clip_norm > 0:
        accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    
    # 优化器更新
    optimizer.step()
    optimizer.zero_grad()
    
    # 学习率调度
    if lr_scheduler is not None:
        lr_scheduler.step()
    
    return loss, output_dict
```

---

## 由浅入深的学习样例

### Sample 1: 数据集探索（入门级）

```python
"""
Sample 1: 探索 LeRobot 数据集
目标: 了解数据集结构和基本访问方式
"""
from pprint import pprint
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

def explore_dataset():
    repo_id = "lerobot/aloha_mobile_cabinet"
    
    # ===== 1. 加载元数据（不下载实际数据） =====
    print("=" * 50)
    print("1. 数据集元数据")
    print("=" * 50)
    
    meta = LeRobotDatasetMetadata(repo_id)
    print(f"总 Episode 数: {meta.total_episodes}")
    print(f"总帧数: {meta.total_frames}")
    print(f"FPS: {meta.fps}")
    print(f"机器人类型: {meta.robot_type}")
    print(f"相机键名: {meta.camera_keys}")
    
    # ===== 2. 查看特征定义 =====
    print("\n" + "=" * 50)
    print("2. 特征定义")
    print("=" * 50)
    pprint(meta.features)
    
    # ===== 3. 查看统计信息 =====
    print("\n" + "=" * 50)
    print("3. 统计信息 (用于归一化)")
    print("=" * 50)
    for key, stats in meta.stats.items():
        if isinstance(stats, dict) and 'mean' in stats:
            print(f"{key}: mean={stats['mean'][:3]}..., std={stats['std'][:3]}...")
    
    # ===== 4. 加载实际数据 =====
    print("\n" + "=" * 50)
    print("4. 加载数据集")
    print("=" * 50)
    
    # 只加载特定 episode
    dataset = LeRobotDataset(repo_id, episodes=[0, 1])
    print(f"选择的 Episode 数: {dataset.num_episodes}")
    print(f"选择的帧数: {dataset.num_frames}")
    
    # ===== 5. 访问单个样本 =====
    print("\n" + "=" * 50)
    print("5. 访问单个样本")
    print("=" * 50)
    
    sample = dataset[0]
    print(f"样本键: {list(sample.keys())}")
    print(f"observation.state 形状: {sample['observation.state'].shape}")
    print(f"action 形状: {sample['action'].shape}")
    
    # 获取图像（如果有）
    for key in meta.camera_keys:
        if key in sample:
            print(f"{key} 形状: {sample[key].shape}")  # (C, H, W)
    
    return dataset

if __name__ == "__main__":
    explore_dataset()
```

### Sample 2: 数据加载与 DataLoader（基础级）

```python
"""
Sample 2: 使用 PyTorch DataLoader 加载数据
目标: 了解如何将数据集用于训练循环
"""
import torch
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def dataloader_example():
    repo_id = "lerobot/aloha_mobile_cabinet"
    
    # ===== 1. 基本数据加载 =====
    dataset = LeRobotDataset(repo_id)
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,  # GPU 训练时加速
    )
    
    # ===== 2. 遍历数据 =====
    for batch in dataloader:
        print(f"Batch observation.state: {batch['observation.state'].shape}")
        print(f"Batch action: {batch['action'].shape}")
        
        # 获取 episode 信息
        print(f"Episode indices: {batch['episode_index'][:5]}")
        break
    
    # ===== 3. 使用时间窗口 (Action Chunking) =====
    print("\n使用时间窗口 (delta_timestamps):")
    
    delta_timestamps = {
        # 历史观测：当前帧 + 前3帧
        "observation.state": [-0.1, -0.05, 0],
        # 未来动作：预测接下来16帧
        "action": [i / dataset.fps for i in range(16)],
    }
    
    dataset_chunked = LeRobotDataset(
        repo_id,
        delta_timestamps=delta_timestamps,
    )
    
    dataloader_chunked = DataLoader(dataset_chunked, batch_size=16, shuffle=True)
    
    for batch in dataloader_chunked:
        print(f"observation.state 形状: {batch['observation.state'].shape}")  # (B, T, D)
        print(f"action 形状: {batch['action'].shape}")  # (B, T, D)
        break

if __name__ == "__main__":
    dataloader_example()
```

### Sample 3: ACT 策略训练（中级）

```python
"""
Sample 3: 训练 ACT (Action Chunking Transformer) 策略
目标: 了解完整的模仿学习训练流程
"""
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def train_act_policy():
    # ===== 配置 =====
    output_dir = Path("outputs/act_training")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_id = "lerobot/aloha_mobile_cabinet"
    
    # ===== 1. 准备数据集 =====
    print("1. 加载数据集元数据...")
    meta = LeRobotDatasetMetadata(dataset_id)
    
    # 将数据集特征转换为策略特征
    features = dataset_to_policy_features(meta.features)
    output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
    input_features = {k: v for k, v in features.items() if k not in output_features}
    
    # ===== 2. 创建策略 =====
    print("2. 创建 ACT 策略...")
    config = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        # ACT 特定参数
        chunk_size=100,           # 动作预测长度
        n_obs_steps=1,            # 观测历史长度
        dim_model=256,            # Transformer 维度
        n_heads=8,                # 注意力头数
        n_encoder_layers=4,       # 编码器层数
        n_decoder_layers=1,       # 解码器层数
    )
    
    policy = ACTPolicy(config)
    policy.to(device)
    policy.train()
    
    # ===== 3. 创建预处理/后处理器 =====
    print("3. 创建处理器...")
    preprocessor, postprocessor = make_pre_post_processors(
        config, 
        dataset_stats=meta.stats
    )
    
    # ===== 4. 准备数据加载器 =====
    print("4. 准备数据加载器...")
    
    # ACT 使用 action chunking，需要设置 delta_timestamps
    delta_timestamps = {
        "action": [i / meta.fps for i in range(config.chunk_size)],
    }
    # 添加图像特征的 delta_timestamps
    for key in config.image_features:
        delta_timestamps[key] = [0]  # 只使用当前帧
    
    dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps)
    
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    
    # ===== 5. 设置优化器 =====
    print("5. 设置优化器...")
    optimizer = config.get_optimizer_preset().build(policy.parameters())
    
    # ===== 6. 训练循环 =====
    print("6. 开始训练...")
    num_epochs = 1
    log_interval = 10
    
    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            # 预处理
            batch = preprocessor(batch)
            
            # 前向传播
            loss, info = policy.forward(batch)
            
            # 反向传播
            loss.backward()
            
            # 优化器更新
            optimizer.step()
            optimizer.zero_grad()
            
            # 日志
            if step % log_interval == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
            
            # 演示目的，只训练几步
            if step >= 50:
                break
    
    # ===== 7. 保存模型 =====
    print("7. 保存模型...")
    policy.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)
    
    print(f"模型已保存到: {output_dir}")
    return policy

if __name__ == "__main__":
    train_act_policy()
```

### Sample 4: 使用命令行训练（中级）

```bash
#!/bin/bash
# Sample 4: 使用 lerobot-train CLI 进行训练

# ===== 基本训练命令 =====
lerobot-train \
  --policy=act \
  --dataset.repo_id=lerobot/aloha_mobile_cabinet \
  --output_dir=outputs/act_aloha \
  --batch_size=8 \
  --steps=100000 \
  --save_freq=10000 \
  --log_freq=100

# ===== 训练 Diffusion Policy =====
lerobot-train \
  --policy=diffusion \
  --dataset.repo_id=lerobot/pusht \
  --output_dir=outputs/diffusion_pusht \
  --batch_size=64 \
  --steps=200000

# ===== 训练 VLA 模型 (SmolVLA) =====
lerobot-train \
  --policy=smolvla \
  --dataset.repo_id=lerobot/aloha_mobile_cabinet \
  --output_dir=outputs/smolvla_aloha \
  --batch_size=4 \
  --steps=50000

# ===== 从检查点恢复训练 =====
lerobot-train \
  --config_path=outputs/act_aloha/checkpoints/last/train_config.json \
  --resume=true

# ===== 使用 wandb 日志 =====
lerobot-train \
  --policy=act \
  --dataset.repo_id=lerobot/aloha_mobile_cabinet \
  --wandb.enable=true \
  --wandb.project=lerobot-training
```

### Sample 5: 策略推理与部署（高级）

```python
"""
Sample 5: 加载预训练策略并进行推理
目标: 了解如何将训练好的策略部署到机器人上
"""
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def inference_example():
    # ===== 1. 加载预训练模型 =====
    print("1. 加载预训练模型...")
    
    model_id = "lerobot/act_aloha_sim_insertion_human"  # Hub 上的模型
    # 或者本地路径: model_id = "outputs/act_training"
    
    policy = ACTPolicy.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    policy.eval()
    
    # ===== 2. 加载处理器 =====
    print("2. 加载处理器...")
    
    # 获取数据集统计信息用于归一化
    dataset_id = "lerobot/aloha_sim_insertion_human"
    meta = LeRobotDatasetMetadata(dataset_id)
    
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=model_id,  # 从预训练路径加载
        dataset_stats=meta.stats,
    )
    
    # ===== 3. 模拟推理循环 =====
    print("3. 推理示例...")
    
    # 构造模拟观测
    batch_size = 1
    observation = {
        "observation.state": torch.randn(batch_size, 14).to(device),  # 关节状态
        "observation.images.top": torch.randn(batch_size, 3, 480, 640).to(device),
    }
    
    # 预处理
    observation = preprocessor(observation)
    
    # 推理
    with torch.no_grad():
        action = policy.select_action(observation)
    
    # 后处理
    action = postprocessor(action)
    
    print(f"输出动作形状: {action['action'].shape}")
    print(f"动作值范围: [{action['action'].min():.3f}, {action['action'].max():.3f}]")
    
    return policy, action


def deploy_to_robot_example():
    """
    部署到真实机器人的示例流程
    注意: 这需要实际的机器人硬件
    """
    from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.policies.utils import build_inference_frame, make_robot_action
    
    # ===== 配置 =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "your-username/your-trained-model"
    dataset_id = "your-username/your-dataset"
    
    # ===== 1. 加载模型 =====
    policy = ACTPolicy.from_pretrained(model_id)
    policy.to(device)
    policy.eval()
    
    # ===== 2. 加载元数据和处理器 =====
    meta = LeRobotDatasetMetadata(dataset_id)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=model_id,
        dataset_stats=meta.stats,
    )
    
    # ===== 3. 配置机器人 =====
    robot_config = SO100FollowerConfig(
        port="/dev/ttyUSB0",  # 根据实际端口修改
        id="follower",
        cameras={
            "front": OpenCVCameraConfig(
                index_or_path=0,
                width=640,
                height=480,
                fps=30
            ),
        }
    )
    
    robot = SO100Follower(robot_config)
    robot.connect()
    
    # ===== 4. 控制循环 =====
    max_steps = 100
    
    try:
        for step in range(max_steps):
            # 获取观测
            obs = robot.get_observation()
            
            # 构建推理帧
            obs_frame = build_inference_frame(
                observation=obs,
                ds_features=meta.features,
                device=device
            )
            
            # 预处理
            obs_frame = preprocessor(obs_frame)
            
            # 推理
            with torch.no_grad():
                action = policy.select_action(obs_frame)
            
            # 后处理
            action = postprocessor(action)
            
            # 转换为机器人动作
            robot_action = make_robot_action(action, meta.features)
            
            # 发送动作
            robot.send_action(robot_action)
            
    finally:
        robot.disconnect()


if __name__ == "__main__":
    inference_example()
```

### Sample 6: 自定义数据集创建（高级）

```python
"""
Sample 6: 创建自定义 LeRobot 数据集
目标: 了解如何将自己的数据转换为 LeRobot 格式
"""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def create_custom_dataset():
    """创建一个简单的自定义数据集"""
    
    repo_id = "my-username/my-custom-dataset"
    root = Path("outputs/custom_dataset")
    
    # ===== 1. 定义数据集特征 =====
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [6],
            "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
        },
        "action": {
            "dtype": "float32",
            "shape": [6],
            "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
        },
        "observation.images.front": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
        },
    }
    
    # ===== 2. 创建数据集 =====
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        features=features,
        root=root,
        robot_type="custom_arm",
        use_videos=True,
    )
    
    # ===== 3. 添加数据 =====
    num_episodes = 3
    frames_per_episode = 100
    
    for episode_idx in range(num_episodes):
        print(f"Recording episode {episode_idx}...")
        
        for frame_idx in range(frames_per_episode):
            # 模拟数据
            frame = {
                "observation.state": np.random.randn(6).astype(np.float32),
                "action": np.random.randn(6).astype(np.float32),
                "observation.images.front": np.random.randint(
                    0, 255, (480, 640, 3), dtype=np.uint8
                ),
                "task": "pick up the object",  # 任务描述
            }
            
            dataset.add_frame(frame)
        
        # 保存 episode
        dataset.save_episode()
    
    # ===== 4. 完成并上传 =====
    dataset.finalize()  # 必须调用！
    
    print(f"数据集已创建: {root}")
    print(f"总帧数: {dataset.num_frames}")
    print(f"总 Episode 数: {dataset.num_episodes}")
    
    # 上传到 Hub（可选）
    # dataset.push_to_hub()
    
    return dataset


if __name__ == "__main__":
    create_custom_dataset()
```

### Sample 7: 流式数据集加载（高级）

```python
"""
Sample 7: 使用流式加载大规模数据集
目标: 了解如何在不下载整个数据集的情况下进行训练
"""
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from torch.utils.data import DataLoader


def streaming_dataset_example():
    # ===== 1. 流式加载数据集 =====
    # 不需要下载整个数据集到本地
    repo_id = "lerobot/aloha_mobile_cabinet"
    
    dataset = StreamingLeRobotDataset(repo_id)
    
    print(f"数据集: {repo_id}")
    print(f"总帧数: {dataset.num_frames}")
    
    # ===== 2. 使用 DataLoader =====
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        num_workers=2,
    )
    
    # ===== 3. 遍历数据 =====
    for batch in dataloader:
        print(f"Batch 形状: {batch['observation.state'].shape}")
        print(f"Action 形状: {batch['action'].shape}")
        break


if __name__ == "__main__":
    streaming_dataset_example()
```

### Sample 8: 多数据集训练（高级）

```python
"""
Sample 8: 使用多个数据集进行联合训练
目标: 了解如何合并多个数据集以提升泛化能力
"""
from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset
from torch.utils.data import DataLoader


def multi_dataset_example():
    # ===== 1. 加载多个数据集 =====
    repo_ids = [
        "lerobot/aloha_mobile_cabinet",
        "lerobot/aloha_mobile_chair",
        "lerobot/aloha_mobile_elevator",
    ]
    
    dataset = MultiLeRobotDataset(repo_ids)
    
    print(f"数据集: {repo_ids}")
    print(f"总帧数: {dataset.num_frames}")
    print(f"总 Episode 数: {dataset.num_episodes}")
    
    # ===== 2. 访问数据 =====
    sample = dataset[0]
    print(f"样本键: {list(sample.keys())}")
    print(f"数据集索引: {sample['dataset_index']}")  # 来自哪个数据集
    
    # ===== 3. 使用 DataLoader =====
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
    )
    
    for batch in dataloader:
        print(f"Batch observation.state: {batch['observation.state'].shape}")
        print(f"Dataset indices: {batch['dataset_index']}")
        break


if __name__ == "__main__":
    multi_dataset_example()
```

---

## 关键概念深入

### 1. Action Chunking (动作分块)

Action Chunking 是 ACT 模型的核心创新，预测一个动作序列而不是单个动作：

```python
# 传统方法: 每步预测一个动作
action_t = policy(observation_t)  # shape: (action_dim,)

# Action Chunking: 预测未来多个动作
action_chunk = policy(observation_t)  # shape: (chunk_size, action_dim)

# 执行时使用时序集成
for t in range(chunk_size):
    execute(action_chunk[t])
```

### 2. 归一化策略

LeRobot 支持多种归一化方式：

```python
# 在 stats.json 中存储的统计信息
{
    "observation.state": {
        "mean": [0.1, 0.2, ...],
        "std": [0.5, 0.3, ...],
        "min": [-1.0, -0.5, ...],
        "max": [1.0, 0.5, ...]
    }
}

# 归一化映射
normalization_mapping = {
    "observation.state": "mean_std",  # z-score 归一化
    "action": "min_max",              # 归一化到 [-1, 1]
    "observation.images.front": None,  # 图像不归一化（由模型处理）
}
```

### 3. 视觉编码器

不同策略使用不同的视觉编码器：

| 策略 | 视觉编码器 |
|------|-----------|
| ACT | ResNet18/34 |
| Diffusion | ResNet + FiLM conditioning |
| SmolVLA | SigLIP Vision Encoder |
| Pi0 | PaliGemma Vision Encoder |
| GR00T | Eagle Vision Encoder |

---

## 常见问题

### Q1: 如何选择合适的策略？

| 场景 | 推荐策略 | 原因 |
|------|---------|------|
| 简单任务，少量数据 | ACT | 训练快，稳定 |
| 复杂任务，多模态 | Diffusion | 表达能力强 |
| 语言条件任务 | SmolVLA / Pi0 | 支持自然语言指令 |
| 大规模预训练 | GR00T | 强大的泛化能力 |

### Q2: 训练不稳定怎么办？

```python
# 1. 降低学习率
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-5)

# 2. 增加梯度裁剪
grad_clip_norm = 1.0
torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)

# 3. 使用更大的 batch size
batch_size = 64  # 或更大

# 4. 检查数据归一化
assert dataset.meta.stats is not None
```

### Q3: 如何处理不同相机配置？

```python
# 使用 rename_map 映射相机名称
rename_map = {
    "observation.images.left_cam": "observation.images.camera1",
    "observation.images.right_cam": "observation.images.camera2",
}

policy = make_policy(
    cfg=policy_config,
    ds_meta=dataset.meta,
    rename_map=rename_map,
)
```

---

## 参考资源

### 官方文档
- [LeRobot 文档](https://huggingface.co/docs/lerobot/index)
- [LeRobotDataset v3.0 文档](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- [Hardware 集成指南](https://huggingface.co/docs/lerobot/integrate_hardware)

### 中文教程
- [LeRobot+SO-ARM101 中文教程 - 同济子豪兄](https://zihao-ai.feishu.cn/wiki/space/7589642043471924447)
- [LeRobot Tutorial CN](https://github.com/CSCSX/LeRobotTutorial-CN)

### 论文
- [ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)
- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)
- [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)

### 数据集
- [Hugging Face LeRobot 数据集](https://huggingface.co/lerobot)
- [探索社区数据集](https://huggingface.co/datasets?other=LeRobot)

### 社区
- [Discord 服务器](https://discord.gg/q8Dzzpym3f)
- [GitHub 仓库](https://github.com/huggingface/lerobot)
