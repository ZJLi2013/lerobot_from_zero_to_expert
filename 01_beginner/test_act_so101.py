"""
测试脚本：Section "ACT + 真实机器人数据训练 Demo"
- 数据集: lerobot/svla_so101_pickplace (真实机器人抓取数据)
- 策略: ACT (Action Chunking Transformer)
- 本脚本仅运行少量步数 (10步) 验证整个训练 pipeline 的正确性
"""
import sys
import traceback
from pathlib import Path

import torch


def make_delta_timestamps(delta_indices, fps: int):
    """将帧索引转换为时间戳"""
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]


def check_env():
    print("=" * 60)
    print("环境检查")
    print("=" * 60)
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
        print("警告: 未检测到 CUDA，将使用 CPU（速度很慢）")
    print()


def test_imports():
    print("=" * 60)
    print("[1/5] 测试导入...")
    print("=" * 60)
    try:
        from lerobot.configs.types import FeatureType
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        from lerobot.datasets.utils import dataset_to_policy_features
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors
        print("  所有模块导入成功 ✓")
        return True
    except ImportError as e:
        print(f"  导入失败: {e}")
        traceback.print_exc()
        return False


def test_dataset_metadata():
    print()
    print("=" * 60)
    print("[2/5] 测试数据集元数据加载...")
    print("=" * 60)
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.utils import dataset_to_policy_features

    dataset_id = "lerobot/svla_so101_pickplace"
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    print(f"  数据集: {dataset_id}")
    print(f"  总帧数: {dataset_metadata.total_frames}")
    print(f"  FPS: {dataset_metadata.fps}")

    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}
    print(f"  输入特征: {list(input_features.keys())}")
    print(f"  输出特征: {list(output_features.keys())}")
    print("  数据集元数据加载成功 ✓")
    return dataset_metadata, input_features, output_features


def test_policy_creation(input_features, output_features):
    print()
    print("=" * 60)
    print("[3/5] 测试 ACT Policy 创建...")
    print("=" * 60)
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    dataset_id = "lerobot/svla_so101_pickplace"
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=100,
        n_obs_steps=1,
        dim_model=256,
        n_heads=8,
        n_encoder_layers=4,
        n_decoder_layers=1,
    )
    policy = ACTPolicy(cfg)
    policy.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  ACT 策略创建成功 ✓")
    print(f"  模型参数量: {n_params:,}")
    print(f"  运行设备: {device}")
    print(f"  chunk_size: {cfg.chunk_size}")
    print(f"  dim_model: {cfg.dim_model}")

    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)
    print(f"  预处理器/后处理器创建成功 ✓")
    return policy, cfg, preprocessor, postprocessor, device


def test_dataloader(cfg):
    print()
    print("=" * 60)
    print("[4/5] 测试数据加载器...")
    print("=" * 60)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    dataset_id = "lerobot/svla_so101_pickplace"
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    fps = dataset_metadata.fps

    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, fps),
    }
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, fps)
        for k in cfg.image_features
    }

    print(f"  action delta_timestamps 数量: {len(delta_timestamps['action'])}")
    dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps)
    print(f"  数据集大小: {len(dataset)} 样本")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,   # 测试用小 batch
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    print(f"  DataLoader 创建成功 ✓  (batch_size=4 for test)")
    return dataloader


def test_training_steps(policy, cfg, preprocessor, dataloader, device, n_steps=5):
    print()
    print("=" * 60)
    print(f"[5/5] 测试训练循环 ({n_steps} 步)...")
    print("=" * 60)
    optimizer = cfg.get_optimizer_preset().build(policy.parameters())

    step = 0
    for batch in dataloader:
        batch = preprocessor(batch)
        loss, info = policy.forward(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_kl = info.get("loss_kl", float("nan"))
        loss_l1 = info.get("loss_l1", float("nan"))
        print(f"  Step {step+1}/{n_steps} | Loss: {loss.item():.4f}  (kl={loss_kl:.4f}, l1={loss_l1:.4f}) ✓")
        step += 1
        if step >= n_steps:
            break

    if torch.cuda.is_available():
        used_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  峰值显存占用: {used_mb:.0f} MB ({used_mb/1024:.2f} GB)")
    print("  训练循环测试通过 ✓")


def test_save(policy, preprocessor, postprocessor):
    print()
    print("=" * 60)
    print("[Extra] 测试模型保存...")
    print("=" * 60)
    output_directory = Path("outputs/test/act_so101_test")
    output_directory.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)
    print(f"  模型已保存到: {output_directory} ✓")


def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   测试: ACT + 真实机器人数据训练 Demo (SO-101 PickPlace) ║")
    print("╚══════════════════════════════════════════════════════════╝")
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

    # 3. 策略创建
    try:
        policy, cfg, preprocessor, postprocessor, device = test_policy_creation(input_features, output_features)
        results["policy_creation"] = True
    except Exception as e:
        print(f"  [FAIL] 策略创建失败: {e}")
        traceback.print_exc()
        results["policy_creation"] = False
        sys.exit(1)

    # 4. 数据加载器
    try:
        dataloader = test_dataloader(cfg)
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

    # Extra: 保存
    try:
        test_save(policy, preprocessor, postprocessor)
        results["save"] = True
    except Exception as e:
        print(f"  [WARN] 模型保存失败: {e}")
        results["save"] = False

    # 汇总
    print()
    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    all_pass = True
    labels = {
        "imports": "模块导入",
        "dataset_metadata": "数据集元数据",
        "policy_creation": "策略创建",
        "dataloader": "数据加载器",
        "training": "训练循环 (5步)",
        "save": "模型保存",
    }
    for key, label in labels.items():
        status = "PASS ✓" if results.get(key) else "FAIL ✗"
        if not results.get(key):
            all_pass = False
        print(f"  {label:<20} {status}")

    print()
    if all_pass:
        print("✓ 所有测试通过！ACT + SO-101 真实机器人数据训练 Pipeline 正常。")
    else:
        print("✗ 部分测试失败，请检查上方错误信息。")
        sys.exit(1)


if __name__ == "__main__":
    main()
