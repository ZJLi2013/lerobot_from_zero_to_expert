"""
LeRobot 数据集可视化 Demo
- 数据集: lerobot/svla_so101_pickplace (真实机器人抓取数据)
- 可视化工具: Rerun SDK

========================================================
三种可视化模式（针对 headless 远端服务器）:

  --mode web    [推荐] 启动 Rerun Web Viewer，浏览器访问
                 远端: python viz_dataset_so101.py --mode web --web-port 9090
                 本地: 浏览器打开 http://<server-ip>:9090

  --mode save   保存 .rrd 文件，拷贝到本地用 rerun 打开
                 远端: python viz_dataset_so101.py --mode save --output-dir outputs/viz
                 本地: scp david@IP:~/github/lerobot/outputs/viz/*.rrd .
                       rerun lerobot_svla_so101_pickplace_episode_0.rrd

  --mode stats  纯统计分析（无需 Rerun），保存 matplotlib 图表
                 适合完全 headless 无图形环境，输出 PNG 到 outputs/viz/

========================================================
"""
import argparse
import gc
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.data


def check_env():
    print("=" * 65)
    print("环境检查")
    print("=" * 65)
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    try:
        import rerun as rr
        print(f"Rerun SDK: {rr.__version__}")
    except ImportError:
        print("Rerun SDK: 未安装（stats 模式无需）")
    print()


def print_dataset_info(dataset):
    """打印数据集基本信息"""
    print("=" * 65)
    print("数据集信息")
    print("=" * 65)
    meta = dataset.meta
    print(f"  数据集 ID:   {dataset.repo_id}")
    print(f"  总帧数:      {meta.total_frames}")
    print(f"  总 Episode:  {meta.total_episodes}")
    print(f"  FPS:         {meta.fps}")
    print(f"  相机列表:    {meta.camera_keys}")
    print(f"  当前 Episode: {dataset.episodes}")
    print()


def visualize_stats(dataset, output_dir: Path):
    """
    模式 stats: 纯统计分析 + matplotlib 图表
    完全 headless，无需 Rerun，输出 PNG。
    """
    print("=" * 65)
    print("[stats 模式] 统计分析 + 保存图表...")
    print("=" * 65)
    import matplotlib
    matplotlib.use("Agg")  # headless backend
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    meta = dataset.meta

    # 加载当前 episode 的全部帧
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=0,
        shuffle=False,
    )

    actions_list = []
    states_list = []
    timestamps = []

    for batch in dataloader:
        if "action" in batch:
            actions_list.append(batch["action"])
        if "observation.state" in batch:
            states_list.append(batch["observation.state"])
        timestamps.extend(batch["timestamp"].tolist())

    # ── 1. Action 维度随时间变化 ──────────────────────────────
    if actions_list:
        actions = torch.cat(actions_list, dim=0).numpy()  # (T, D)
        n_dims = actions.shape[1]
        fig, axes = plt.subplots(n_dims, 1, figsize=(14, 2 * n_dims), sharex=True)
        if n_dims == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax.plot(timestamps, actions[:, i], linewidth=0.8, color=f"C{i}")
            ax.set_ylabel(f"action[{i}]", fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("timestamp (s)")
        fig.suptitle(
            f"{dataset.repo_id} | Episode {dataset.episodes[0]} | Action Dimensions",
            fontsize=11,
        )
        plt.tight_layout()
        ep = dataset.episodes[0]
        save_path = output_dir / f"action_episode_{ep}.png"
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"  Action 图表已保存: {save_path}")

    # ── 2. State 维度随时间变化 ───────────────────────────────
    if states_list:
        states = torch.cat(states_list, dim=0).numpy()
        n_dims = states.shape[1]
        fig, axes = plt.subplots(n_dims, 1, figsize=(14, 2 * n_dims), sharex=True)
        if n_dims == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax.plot(timestamps, states[:, i], linewidth=0.8, color=f"C{i}")
            ax.set_ylabel(f"state[{i}]", fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("timestamp (s)")
        fig.suptitle(
            f"{dataset.repo_id} | Episode {dataset.episodes[0]} | State Dimensions",
            fontsize=11,
        )
        plt.tight_layout()
        save_path = output_dir / f"state_episode_{ep}.png"
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"  State  图表已保存: {save_path}")

    # ── 3. 抽帧截图（每5秒一帧）──────────────────────────────
    camera_keys = meta.camera_keys
    if camera_keys:
        # 找到时间步最接近 0s, 5s, 10s ... 的帧
        ts_arr = np.array(timestamps)
        n_frames = len(ts_arr)
        duration = ts_arr[-1] - ts_arr[0] if n_frames > 1 else 0
        sample_times = np.arange(0, duration + 1e-6, 5.0)

        fig, axes = plt.subplots(
            len(camera_keys), len(sample_times),
            figsize=(3 * len(sample_times), 3 * len(camera_keys)),
        )
        if len(camera_keys) == 1:
            axes = [axes]
        if len(sample_times) == 1:
            axes = [[ax] for ax in axes]

        for ci, cam_key in enumerate(camera_keys):
            for ti, t_target in enumerate(sample_times):
                # 找到最近帧索引
                frame_idx = int(np.argmin(np.abs(ts_arr - (ts_arr[0] + t_target))))
                sample = dataset[frame_idx]
                img = sample[cam_key]  # (C, H, W) float32
                img_np = (img * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
                axes[ci][ti].imshow(img_np)
                axes[ci][ti].axis("off")
                if ci == 0:
                    axes[ci][ti].set_title(f"t≈{ts_arr[0]+t_target:.1f}s", fontsize=8)
                if ti == 0:
                    axes[ci][ti].set_ylabel(cam_key.split(".")[-1], fontsize=8)

        fig.suptitle(
            f"{dataset.repo_id} | Episode {dataset.episodes[0]} | Camera Frames",
            fontsize=11,
        )
        plt.tight_layout()
        save_path = output_dir / f"frames_episode_{ep}.png"
        plt.savefig(save_path, dpi=100)
        plt.close()
        print(f"  相机截图已保存: {save_path}")

    # ── 4. 打印统计摘要 ───────────────────────────────────────
    print()
    print("  统计摘要")
    print(f"  {'帧数':<15} {len(timestamps)}")
    print(f"  {'时长':<15} {timestamps[-1] - timestamps[0]:.2f} s")
    if actions_list:
        a = torch.cat(actions_list, dim=0)
        print(f"  {'Action 维度':<15} {a.shape[1]}")
        print(f"  {'Action 均值':<15} {a.mean(0).numpy().round(4)}")
        print(f"  {'Action 标准差':<15} {a.std(0).numpy().round(4)}")
    if states_list:
        s = torch.cat(states_list, dim=0)
        print(f"  {'State 维度':<15} {s.shape[1]}")
        print(f"  {'State 均值':<15} {s.mean(0).numpy().round(4)}")
        print(f"  {'State 标准差':<15} {s.std(0).numpy().round(4)}")
    print()
    print(f"  所有图表已保存至 {output_dir}/")


def visualize_rerun(
    dataset,
    mode: str,
    web_port: int,
    grpc_port: int,
    save: bool,
    output_dir: Path,
):
    """
    使用 Rerun SDK 可视化（web 模式 或 save 模式）
    """
    import rerun as rr
    from lerobot.utils.constants import ACTION, OBS_STATE

    def to_hwc_uint8(chw: torch.Tensor) -> np.ndarray:
        return (chw * 255).to(torch.uint8).permute(1, 2, 0).numpy()

    episode_index = dataset.episodes[0]
    repo_id = dataset.repo_id

    print("=" * 65)
    print(f"[{mode} 模式] 启动 Rerun 可视化...")
    print("=" * 65)

    spawn_local = (mode == "local") and not save
    rr.init(f"{repo_id}/episode_{episode_index}", spawn=spawn_local)
    gc.collect()

    if mode == "web":
        server_uri = rr.serve_grpc(grpc_port=grpc_port)
        rr.serve_web_viewer(open_browser=False, web_port=web_port, connect_to=server_uri)
        print(f"  ✓ Rerun Web Viewer 已启动")
        print(f"  ✓ 浏览器访问:  http://<server-ip>:{web_port}")
        print(f"  ✓ 桌面 rerun:  rerun rerun+http://<server-ip>:{grpc_port}/proxy")
        print(f"  ✓ SSH 转发后:  http://localhost:{web_port}")
        print()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        num_workers=0,
        shuffle=False,
    )

    import tqdm
    first_index = None
    total = len(dataloader)
    print(f"  正在向 Rerun 发送数据（{total} 批次）...")

    for batch in tqdm.tqdm(dataloader, total=total):
        if first_index is None:
            first_index = batch["index"][0].item()
        for i in range(len(batch["index"])):
            rr.set_time("frame_index", sequence=batch["index"][i].item() - first_index)
            rr.set_time("timestamp", timestamp=batch["timestamp"][i].item())

            for key in dataset.meta.camera_keys:
                if key in batch:
                    rr.log(key, entity=rr.Image(to_hwc_uint8(batch[key][i])))

            if ACTION in batch:
                for d, val in enumerate(batch[ACTION][i]):
                    rr.log(f"action/{d}", rr.Scalars(val.item()))

            if OBS_STATE in batch:
                for d, val in enumerate(batch[OBS_STATE][i]):
                    rr.log(f"state/{d}", rr.Scalars(val.item()))

    if mode == "local" and save:
        output_dir.mkdir(parents=True, exist_ok=True)
        rid = repo_id.replace("/", "_")
        rrd_path = output_dir / f"{rid}_episode_{episode_index}.rrd"
        rr.save(rrd_path)
        print(f"  ✓ .rrd 文件已保存: {rrd_path}")
        print()
        print("  本地打开方式（在本地机器上运行）:")
        print(f"    scp david@<server-ip>:~/github/lerobot/{rrd_path} .")
        print(f"    rerun {rrd_path.name}")

    elif mode == "web":
        print()
        print("  数据发送完毕，Web Viewer 保持运行中...")
        print("  按 Ctrl+C 停止服务。")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("  已停止。")


def main():
    parser = argparse.ArgumentParser(
        description="LeRobot 数据集可视化 Demo (svla_so101_pickplace)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # Web 浏览器模式（headless 服务器推荐）
  python viz_dataset_so101.py --mode web --web-port 9090

  # 保存 .rrd 文件
  python viz_dataset_so101.py --mode save --output-dir outputs/viz

  # 纯统计模式（无需 Rerun）
  python viz_dataset_so101.py --mode stats --output-dir outputs/viz

  # 多 episode 统计
  python viz_dataset_so101.py --mode stats --episode-index 0 1 2
        """,
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="lerobot/svla_so101_pickplace",
        help="HuggingFace 数据集 ID",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        nargs="+",
        default=[0],
        help="要可视化的 episode 编号（可多个，stats 模式支持批量）",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="stats",
        choices=["web", "save", "stats"],
        help=(
            "可视化模式:\n"
            "  web   - Rerun Web Viewer，浏览器访问（headless 推荐）\n"
            "  save  - 保存 .rrd 文件，拷贝到本地用 rerun 打开\n"
            "  stats - 纯统计+matplotlib，完全 headless"
        ),
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=9090,
        help="Web Viewer 端口（mode=web 时）",
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=9876,
        help="gRPC 端口（mode=web 时）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/viz"),
        help="输出目录（mode=save 或 stats 时）",
    )
    args = parser.parse_args()

    print()
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║       LeRobot 数据集可视化 Demo                               ║")
    print(f"║       数据集: {args.repo_id:<47}║")
    print(f"║       模式:   {args.mode:<47}║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()

    check_env()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if args.mode == "stats":
        # stats 模式支持批量 episode
        for ep_idx in args.episode_index:
            print(f"── Episode {ep_idx} ────────────────────────────────────────────")
            dataset = LeRobotDataset(args.repo_id, episodes=[ep_idx])
            print_dataset_info(dataset)
            visualize_stats(dataset, args.output_dir)
    else:
        # web / save 模式只处理第一个 episode
        ep_idx = args.episode_index[0]
        dataset = LeRobotDataset(args.repo_id, episodes=[ep_idx])
        print_dataset_info(dataset)
        save = (args.mode == "save")
        visualize_rerun(
            dataset=dataset,
            mode="local" if save else "web",
            web_port=args.web_port,
            grpc_port=args.grpc_port,
            save=save,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
