"""
SDG 合成数据 Rerun 可视化
=========================
直接读取 2_basic_collect.py 输出的 npy 文件，生成 .rrd 文件。
本地用 `rerun <file>.rrd` 打开即可同步查看：
  - 双相机图像 (up / side)
  - 6 维 state 曲线
  - 6 维 action 曲线
  - state vs action 对比

用法：
  python viz_sdg_rerun.py --input /output/npy --output /output/viz
  # 生成: /output/viz/so101_sdg_episode_0.rrd ...

  # 本地查看：
  rerun so101_sdg_episode_0.rrd
"""
import argparse
from pathlib import Path

import numpy as np

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def parse_args():
    p = argparse.ArgumentParser(description="SDG npy → Rerun .rrd")
    p.add_argument("--input", required=True, help="npy 数据目录")
    p.add_argument("--output", default="/tmp/viz_sdg", help=".rrd 输出目录")
    p.add_argument("--episodes", type=int, nargs="*", default=None,
                   help="要可视化的 episode 编号（默认全部）")
    p.add_argument("--fps", type=int, default=30)
    return p.parse_args()


def load_data(input_dir: Path):
    data = {}
    for name in ["states", "actions", "images_up", "images_side",
                  "timestamps", "frame_indices", "episode_indices"]:
        p = input_dir / f"{name}.npy"
        arr = np.load(str(p))
        data[name] = arr
        print(f"  {name}: {arr.shape} {arr.dtype}")
    return data


def save_episode_rrd(data: dict, ep_idx: int, output_dir: Path, fps: int):
    """Generate .rrd for a single episode."""
    import rerun as rr

    mask = data["episode_indices"] == ep_idx
    states = data["states"][mask]
    actions = data["actions"][mask]
    imgs_up = data["images_up"][mask]
    imgs_side = data["images_side"][mask]
    timestamps = data["timestamps"][mask]
    frame_indices = data["frame_indices"][mask]

    n_frames = len(states)
    n_joints = min(states.shape[1], len(JOINT_NAMES))

    rrd_name = f"so101_sdg_episode_{ep_idx}"
    rrd_path = output_dir / f"{rrd_name}.rrd"

    rr.init(f"so101_genesis_sdg/episode_{ep_idx}", spawn=False)

    print(f"  episode {ep_idx}: {n_frames} frames → {rrd_path}")

    for i in range(n_frames):
        t = float(timestamps[i])
        fi = int(frame_indices[i])

        rr.set_time("frame_index", sequence=fi)
        rr.set_time("timestamp", timestamp=t)

        # Images
        rr.log("observation.images.up", rr.Image(imgs_up[i]))
        rr.log("observation.images.side", rr.Image(imgs_side[i]))

        # State per joint
        for j in range(n_joints):
            jname = JOINT_NAMES[j]
            rr.log(f"state/{jname}", rr.Scalars(float(states[i, j])))

        # Action per joint
        for j in range(n_joints):
            jname = JOINT_NAMES[j]
            rr.log(f"action/{jname}", rr.Scalars(float(actions[i, j])))

        # State vs Action overlay (tracking error)
        for j in range(n_joints):
            jname = JOINT_NAMES[j]
            rr.log(f"tracking/{jname}/state", rr.Scalars(float(states[i, j])))
            rr.log(f"tracking/{jname}/action", rr.Scalars(float(actions[i, j])))

    rr.save(str(rrd_path))
    size_mb = rrd_path.stat().st_size / (1024 * 1024)
    print(f"  saved: {rrd_path} ({size_mb:.1f} MB)")
    return rrd_path


def main():
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  SDG 合成数据 → Rerun .rrd")
    print(f"{'='*60}")

    print("\n[1/2] 加载数据 ...")
    data = load_data(input_dir)

    unique_eps = np.unique(data["episode_indices"])
    if args.episodes is not None:
        unique_eps = [e for e in unique_eps if e in args.episodes]

    print(f"\n[2/2] 生成 .rrd 文件 ({len(unique_eps)} episodes) ...")
    rrd_files = []
    for ep_idx in unique_eps:
        rrd = save_episode_rrd(data, int(ep_idx), output_dir, args.fps)
        rrd_files.append(rrd)

    print(f"\n{'='*60}")
    print(f"  完成！共生成 {len(rrd_files)} 个 .rrd 文件")
    print(f"{'='*60}")
    for f in rrd_files:
        print(f"  {f}")
    print(f"\n  本地查看：rerun <file>.rrd\n")


if __name__ == "__main__":
    main()
