"""
测试脚本：Section "快速入门：单卡训练 Demo (RTX 4090)"
- 数据集: lerobot/pusht (2D 推箱子任务)
- 策略: Diffusion Policy
- 本脚本仅运行少量步数 (10步) 验证整个训练 pipeline 的正确性
"""
import sys
import traceback
from pathlib import Path

import torch


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
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
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

    dataset_metadata = LeRobotDatasetMetadata("lerobot/pusht")
    print(f"  数据集: lerobot/pusht")
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
    print("[3/5] 测试 Diffusion Policy 创建...")
    print("=" * 60)
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    dataset_metadata = LeRobotDatasetMetadata("lerobot/pusht")
    cfg = DiffusionConfig(input_features=input_features, output_features=output_features)
    policy = DiffusionPolicy(cfg)
    policy.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  策略创建成功 ✓")
    print(f"  模型参数量: {n_params:,}")
    print(f"  运行设备: {device}")

    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)
    print(f"  预处理器/后处理器创建成功 ✓")
    return policy, cfg, preprocessor, device


def test_dataloader():
    print()
    print("=" * 60)
    print("[4/5] 测试数据加载器...")
    print("=" * 60)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    delta_timestamps = {
        "observation.image": [-0.1, 0.0],
        "observation.state": [-0.1, 0.0],
        "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    }
    dataset = LeRobotDataset("lerobot/pusht", delta_timestamps=delta_timestamps)
    print(f"  数据集大小: {len(dataset)} 样本")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=2,
        batch_size=8,   # 测试用小 batch
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    print(f"  DataLoader 创建成功 ✓  (batch_size=8 for test)")
    return dataloader


def test_training_steps(policy, cfg, preprocessor, dataloader, device, n_steps=5):
    print()
    print("=" * 60)
    print(f"[5/5] 测试训练循环 ({n_steps} 步)...")
    print("=" * 60)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    step = 0
    for batch in dataloader:
        batch = preprocessor(batch)
        loss, _ = policy.forward(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"  Step {step+1}/{n_steps} | Loss: {loss.item():.4f} ✓")
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
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors

    output_directory = Path("outputs/test/pusht_diffusion_test")
    output_directory.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)
    print(f"  模型已保存到: {output_directory} ✓")


def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  测试: 快速入门·单卡训练 Demo — Diffusion Policy + PushT ║")
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
        policy, cfg, preprocessor, device = test_policy_creation(input_features, output_features)
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.policies.factory import make_pre_post_processors
        _, postprocessor = make_pre_post_processors(cfg, dataset_stats=LeRobotDatasetMetadata("lerobot/pusht").stats)
        results["policy_creation"] = True
    except Exception as e:
        print(f"  [FAIL] 策略创建失败: {e}")
        traceback.print_exc()
        results["policy_creation"] = False
        sys.exit(1)

    # 4. 数据加载器
    try:
        dataloader = test_dataloader()
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
        print("✓ 所有测试通过！Diffusion Policy + PushT 训练 Pipeline 正常。")
    else:
        print("✗ 部分测试失败，请检查上方错误信息。")
        sys.exit(1)


if __name__ == "__main__":
    main()
