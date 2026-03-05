"""
测试脚本：SmolVLA VLA 模型微调训练 Demo
- 数据集: lerobot/svla_so101_pickplace (真实机器人抓取数据，含语言描述)
- 策略: SmolVLA (450M VLA = SmolVLM2-500M 视觉语言骨干 + Flow Matching 动作专家)
- 预训练权重: lerobot/smolvla_base (Hugging Face Hub)
- 微调策略: freeze_vision_encoder=True, train_expert_only=True
- 本脚本仅运行少量步数 (5步) 验证整个训练 pipeline 的正确性

参考: https://docs.phospho.ai/learn/train-smolvla
     https://huggingface.co/lerobot/smolvla_base
"""
import sys
import time
import traceback
from pathlib import Path

import torch


def make_delta_timestamps(delta_indices, fps: int):
    """将帧索引转换为时间戳"""
    if delta_indices is None:
        return [0.0]
    return [i / fps for i in delta_indices]


def check_env():
    print("=" * 65)
    print("环境检查")
    print("=" * 65)
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA 可用: {cuda_ok}")
    if cuda_ok:
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU[{i}]: {props.name}  显存: {props.total_memory / 1024**3:.1f} GB")
    else:
        print("警告: 未检测到 CUDA，将使用 CPU（速度极慢）")
    print()


def test_imports():
    print("=" * 65)
    print("[1/6] 测试导入...")
    print("=" * 65)
    try:
        from lerobot.configs.types import FeatureType
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        from lerobot.datasets.utils import dataset_to_policy_features
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        print("  所有模块导入成功 ✓")
        return True
    except ImportError as e:
        print(f"  导入失败: {e}")
        traceback.print_exc()
        return False


def test_dataset_metadata():
    print()
    print("=" * 65)
    print("[2/6] 测试数据集元数据加载...")
    print("=" * 65)
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.utils import dataset_to_policy_features

    dataset_id = "lerobot/svla_so101_pickplace"
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    print(f"  数据集: {dataset_id}")
    print(f"  总帧数: {dataset_metadata.total_frames}")
    print(f"  FPS: {dataset_metadata.fps}")
    print(f"  Episode 数: {dataset_metadata.total_episodes}")

    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}
    print(f"  输入特征: {list(input_features.keys())}")
    print(f"  输出特征: {list(output_features.keys())}")

    # 检查是否有语言任务描述（tasks 可能是 dict 或 DataFrame）
    tasks = dataset_metadata.tasks
    try:
        import pandas as pd
        if isinstance(tasks, pd.DataFrame):
            sample_task = tasks.iloc[0]["task"] if not tasks.empty else "(无)"
        elif isinstance(tasks, dict):
            sample_task = list(tasks.values())[0] if tasks else "(无)"
        else:
            sample_task = str(tasks)
    except Exception:
        sample_task = "(无法读取)"
    print(f"  语言任务描述示例: '{sample_task}'")
    print("  数据集元数据加载成功 ✓")
    return dataset_metadata, input_features, output_features


def test_policy_creation(input_features, output_features, dataset_metadata):
    print()
    print("=" * 65)
    print("[3/6] 测试 SmolVLA Policy 创建 (从预训练权重加载)...")
    print("=" * 65)
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    # 创建 SmolVLAConfig，使用数据集的输入/输出特征
    # freeze_vision_encoder=True, train_expert_only=True 为默认微调设置
    cfg = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=50,
        n_action_steps=50,
        freeze_vision_encoder=True,
        train_expert_only=True,
        train_state_proj=True,
    )

    print(f"  SmolVLAConfig 创建成功 ✓")
    print(f"  vlm_model_name: {cfg.vlm_model_name}")
    print(f"  chunk_size: {cfg.chunk_size}")
    print(f"  max_state_dim: {cfg.max_state_dim}")
    print(f"  max_action_dim: {cfg.max_action_dim}")
    print(f"  freeze_vision_encoder: {cfg.freeze_vision_encoder}")
    print(f"  train_expert_only: {cfg.train_expert_only}")

    # 从预训练权重加载 SmolVLA base（HuggingFace Hub）
    # 注意：strict=False 允许特征维度不完全匹配（状态/动作投影层会重新初始化）
    print(f"  正在从 lerobot/smolvla_base 加载预训练权重...")
    t0 = time.time()
    policy = SmolVLAPolicy.from_pretrained(
        "lerobot/smolvla_base",
        config=cfg,
        strict=False,
    )
    load_time = time.time() - t0
    policy.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)

    # 统计可训练参数 vs 冻结参数
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"  SmolVLA Policy 加载成功 ✓  (耗时: {load_time:.1f}s)")
    print(f"  总参数量: {total_params:,} (~{total_params/1e6:.0f}M)")
    print(f"  可训练参数: {trainable_params:,} (~{trainable_params/1e6:.1f}M)")
    print(f"  冻结参数: {frozen_params:,} (~{frozen_params/1e6:.0f}M)")
    print(f"  运行设备: {device}")

    # 创建预处理器/后处理器（从预训练路径加载以保持兼容性）
    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path="lerobot/smolvla_base",
        dataset_stats=dataset_metadata.stats,
    )
    print(f"  预处理器/后处理器创建成功 ✓")
    return policy, cfg, preprocessor, postprocessor, device


def test_dataloader(cfg, dataset_metadata):
    print()
    print("=" * 65)
    print("[4/6] 测试数据加载器...")
    print("=" * 65)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_id = "lerobot/svla_so101_pickplace"
    fps = dataset_metadata.fps

    # SmolVLA: 当前帧观测 + chunk_size 步动作序列
    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, fps),
    }
    # 所有图像特征取当前帧
    for img_key in cfg.image_features:
        delta_timestamps[img_key] = make_delta_timestamps(cfg.observation_delta_indices, fps)
    # 状态取当前帧
    delta_timestamps["observation.state"] = make_delta_timestamps(cfg.observation_delta_indices, fps)

    print(f"  action chunk 长度: {len(delta_timestamps['action'])} 步 ({cfg.chunk_size} 步)")
    print(f"  期望图像特征: {list(cfg.image_features.keys())}")

    dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps)
    print(f"  数据集大小: {len(dataset)} 样本")

    # SmolVLA 推荐 batch_size=64 (论文)，测试用 batch_size=2
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    print(f"  DataLoader 创建成功 ✓  (batch_size=2 for test)")
    return dataloader


def test_training_steps(policy, cfg, preprocessor, dataloader, device, n_steps=5):
    print()
    print("=" * 65)
    print(f"[5/6] 测试训练循环 ({n_steps} 步)...")
    print("=" * 65)
    # 只优化可训练参数（action expert + state proj，不含 frozen VLM）
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.optimizer_lr,
        betas=cfg.optimizer_betas,
        eps=cfg.optimizer_eps,
        weight_decay=cfg.optimizer_weight_decay,
    )

    step = 0
    for batch in dataloader:
        t0 = time.time()
        batch = preprocessor(batch)
        loss, info = policy.forward(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, cfg.optimizer_grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad()
        step_time = time.time() - t0

        print(
            f"  Step {step+1}/{n_steps} | Loss: {loss.item():.4f} | "
            f"耗时: {step_time:.2f}s ✓"
        )
        step += 1
        if step >= n_steps:
            break

    if torch.cuda.is_available():
        used_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  峰值显存占用: {used_mb:.0f} MB ({used_mb/1024:.2f} GB)")
    print("  训练循环测试通过 ✓")


def test_save(policy, preprocessor, postprocessor):
    print()
    print("=" * 65)
    print("[6/6] 测试模型保存...")
    print("=" * 65)
    output_directory = Path("outputs/test/smolvla_so101_test")
    output_directory.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)
    print(f"  模型已保存到: {output_directory} ✓")


def main():
    print()
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║   测试: SmolVLA VLA 模型微调训练 Demo (SO-101 PickPlace)      ║")
    print("║   预训练权重: lerobot/smolvla_base                            ║")
    print("║   数据集:    lerobot/svla_so101_pickplace                     ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()

    check_env()

    results = {}

    # 1. 导入
    results["imports"] = test_imports()
    if not results["imports"]:
        print("\n[FAIL] 导入失败，终止测试。")
        sys.exit(1)

    # 2. 数据集元数据
    try:
        dataset_metadata, input_features, output_features = test_dataset_metadata()
        results["dataset_metadata"] = True
    except Exception as e:
        print(f"  [FAIL] 数据集元数据加载失败: {e}")
        traceback.print_exc()
        results["dataset_metadata"] = False
        sys.exit(1)

    # 3. 策略创建（从预训练权重加载）
    try:
        policy, cfg, preprocessor, postprocessor, device = test_policy_creation(
            input_features, output_features, dataset_metadata
        )
        results["policy_creation"] = True
    except Exception as e:
        print(f"  [FAIL] 策略创建失败: {e}")
        traceback.print_exc()
        results["policy_creation"] = False
        sys.exit(1)

    # 4. 数据加载器
    try:
        dataloader = test_dataloader(cfg, dataset_metadata)
        results["dataloader"] = True
    except Exception as e:
        print(f"  [FAIL] 数据加载器创建失败: {e}")
        traceback.print_exc()
        results["dataloader"] = False
        sys.exit(1)

    # 5. 训练循环
    try:
        test_training_steps(policy, cfg, preprocessor, dataloader, device, n_steps=5)
        results["training"] = True
    except Exception as e:
        print(f"  [FAIL] 训练循环失败: {e}")
        traceback.print_exc()
        results["training"] = False
        sys.exit(1)

    # 6. 模型保存
    try:
        test_save(policy, preprocessor, postprocessor)
        results["save"] = True
    except Exception as e:
        print(f"  [WARN] 模型保存失败: {e}")
        results["save"] = False

    # 汇总
    print()
    print("=" * 65)
    print("测试结果汇总")
    print("=" * 65)
    all_pass = True
    labels = {
        "imports": "模块导入",
        "dataset_metadata": "数据集元数据",
        "policy_creation": "SmolVLA策略创建(pretrained)",
        "dataloader": "数据加载器",
        "training": "训练循环 (5步)",
        "save": "模型保存",
    }
    for key, label in labels.items():
        status = "PASS ✓" if results.get(key) else "FAIL ✗"
        if not results.get(key):
            all_pass = False
        print(f"  {label:<30} {status}")

    print()
    if all_pass:
        print("✓ 所有测试通过！SmolVLA 微调训练 Pipeline 正常。")
    else:
        print("✗ 部分测试失败，请检查上方错误信息。")
        sys.exit(1)


if __name__ == "__main__":
    main()
