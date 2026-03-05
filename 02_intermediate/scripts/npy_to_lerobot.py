"""
将 sdg_so101_genesis.py 输出的 npy 文件转为 LeRobot v3 数据集格式。

输出结构（与 svla_so101_pickplace 同构）：
  <output>/
    data/chunk-000/file-000.parquet
    videos/observation.images.up/chunk-000/file-000.mp4
    videos/observation.images.side/chunk-000/file-000.mp4
    meta/info.json
    meta/episodes/chunk-000/file-000.parquet
    meta/tasks/chunk-000/file-000.parquet

用法：
  python npy_to_lerobot.py --input /tmp/sdg_so101_output --output /tmp/lerobot_so101_sim
  python npy_to_lerobot.py --input /tmp/sdg_so101_output --output /tmp/lerobot_so101_sim --fps 30
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[warn] opencv not found, will save frames as png fallback")

# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="npy → LeRobot dataset converter")
    p.add_argument("--input",  required=True, help="sdg_so101_genesis.py 输出目录")
    p.add_argument("--output", required=True, help="LeRobot 数据集输出目录")
    p.add_argument("--fps",    type=int, default=30)
    p.add_argument("--task",   default="pick up the red cube and place it to the right")
    p.add_argument("--repo-id", default="local/so101-genesis-sim")
    return p.parse_args()


def load_npy(input_dir: Path):
    """Load all npy files and return a dict."""
    data = {}
    for name in ["states", "actions", "images_up", "images_side",
                  "timestamps", "frame_indices", "episode_indices"]:
        p = input_dir / f"{name}.npy"
        if p.exists():
            data[name] = np.load(str(p))
            print(f"  loaded {name}: {data[name].shape}")
        else:
            raise FileNotFoundError(f"Missing: {p}")
    return data


def write_data_parquet(data: dict, output_dir: Path, task_str: str):
    """Write data/chunk-000/file-000.parquet with one row per frame."""
    chunk_dir = output_dir / "data" / "chunk-000"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    n = len(data["states"])
    episode_indices = data["episode_indices"].astype(np.int64)
    frame_indices = data["frame_indices"].astype(np.int64)
    timestamps = data["timestamps"].astype(np.float32)

    # global index
    global_idx = np.arange(n, dtype=np.int64)

    # task_index: all same task for now
    task_indices = np.zeros(n, dtype=np.int64)

    # Build per-column arrays for state and action (list of float32 arrays)
    states_list = [row.tolist() for row in data["states"]]
    actions_list = [row.tolist() for row in data["actions"]]

    table = pa.table({
        "observation.state": pa.array(states_list, type=pa.list_(pa.float32())),
        "action":            pa.array(actions_list, type=pa.list_(pa.float32())),
        "timestamp":         pa.array(timestamps, type=pa.float32()),
        "frame_index":       pa.array(frame_indices, type=pa.int64()),
        "episode_index":     pa.array(episode_indices, type=pa.int64()),
        "index":             pa.array(global_idx, type=pa.int64()),
        "task_index":        pa.array(task_indices, type=pa.int64()),
    })

    out_path = chunk_dir / "file-000.parquet"
    pq.write_table(table, str(out_path))
    print(f"  parquet: {out_path} ({n} rows)")
    return n


def write_videos(data: dict, output_dir: Path, fps: int):
    """Write MP4 videos per camera, one file per episode chunk."""
    for cam_key, npy_key in [("observation.images.up", "images_up"),
                              ("observation.images.side", "images_side")]:
        vid_dir = output_dir / "videos" / cam_key / "chunk-000"
        vid_dir.mkdir(parents=True, exist_ok=True)

        images = data[npy_key]  # (N, H, W, 3)
        ep_ids = data["episode_indices"]
        unique_eps = np.unique(ep_ids)

        for ep_idx in unique_eps:
            mask = ep_ids == ep_idx
            ep_frames = images[mask]  # (T, H, W, 3)
            h, w = ep_frames.shape[1], ep_frames.shape[2]

            mp4_path = vid_dir / f"episode_{int(ep_idx):06d}.mp4"

            if HAS_CV2:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(mp4_path), fourcc, fps, (w, h))
                for frame in ep_frames:
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    writer.write(bgr)
                writer.release()
            else:
                # Fallback: save frames as PNGs in a subdirectory
                frames_dir = vid_dir / f"episode_{int(ep_idx):06d}"
                frames_dir.mkdir(exist_ok=True)
                from PIL import Image
                for i, frame in enumerate(ep_frames):
                    Image.fromarray(frame).save(str(frames_dir / f"{i:06d}.png"))

            print(f"  video: {mp4_path} ({len(ep_frames)} frames, {w}x{h})")


def write_episodes_parquet(data: dict, output_dir: Path):
    """Write meta/episodes/chunk-000/file-000.parquet with episode boundaries."""
    ep_dir = output_dir / "meta" / "episodes" / "chunk-000"
    ep_dir.mkdir(parents=True, exist_ok=True)

    ep_ids = data["episode_indices"]
    unique_eps = np.unique(ep_ids)

    rows = []
    global_offset = 0
    for ep_idx in unique_eps:
        mask = ep_ids == ep_idx
        n_frames = int(mask.sum())
        rows.append({
            "episode_index": int(ep_idx),
            "length": n_frames,
            "index_from": global_offset,
            "index_to": global_offset + n_frames,
        })
        global_offset += n_frames

    df = pd.DataFrame(rows)
    out_path = ep_dir / "file-000.parquet"
    df.to_parquet(str(out_path), index=False)
    print(f"  episodes: {out_path} ({len(rows)} episodes)")
    return rows


def write_tasks_parquet(output_dir: Path, task_str: str):
    """Write meta/tasks/chunk-000/file-000.parquet."""
    task_dir = output_dir / "meta" / "tasks" / "chunk-000"
    task_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([{"task_index": 0, "task": task_str}])
    out_path = task_dir / "file-000.parquet"
    df.to_parquet(str(out_path), index=False)
    print(f"  tasks: {out_path}")


def write_info_json(output_dir: Path, fps: int, n_frames: int,
                    n_episodes: int, task_str: str, repo_id: str,
                    state_dim: int, img_h: int, img_w: int):
    """Write meta/info.json matching LeRobot v3 schema."""
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "codebase_version": "v3.0",
        "robot_type": "so101",
        "total_episodes": n_episodes,
        "total_frames": n_frames,
        "fps": fps,
        "repo_id": repo_id,
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [state_dim],
                "names": [
                    "shoulder_pan.pos",
                    "shoulder_lift.pos",
                    "elbow_flex.pos",
                    "wrist_flex.pos",
                    "wrist_roll.pos",
                    "gripper.pos",
                ][:state_dim],
            },
            "action": {
                "dtype": "float32",
                "shape": [state_dim],
                "names": [
                    "shoulder_pan.pos",
                    "shoulder_lift.pos",
                    "elbow_flex.pos",
                    "wrist_flex.pos",
                    "wrist_roll.pos",
                    "gripper.pos",
                ][:state_dim],
            },
            "observation.images.up": {
                "dtype": "video",
                "shape": [3, img_h, img_w],
                "names": ["channel", "height", "width"],
                "video_info": {
                    "video.fps": fps,
                    "video.codec": "mp4v",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.images.side": {
                "dtype": "video",
                "shape": [3, img_h, img_w],
                "names": ["channel", "height", "width"],
                "video_info": {
                    "video.fps": fps,
                    "video.codec": "mp4v",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
        "task_to_task_index": {task_str: 0},
    }

    out_path = meta_dir / "info.json"
    with open(str(out_path), "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"  info.json: {out_path}")


def main():
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  npy → LeRobot 格式转换")
    print(f"{'='*60}")
    print(f"  input:  {input_dir}")
    print(f"  output: {output_dir}")
    print(f"  fps:    {args.fps}")
    print()

    # 1. Load
    print("[1/5] 加载 npy 数据 ...")
    data = load_npy(input_dir)

    # 2. Write parquet
    print("\n[2/5] 写入 data parquet ...")
    n_frames = write_data_parquet(data, output_dir, args.task)

    # 3. Write videos
    print("\n[3/5] 编码 MP4 视频 ...")
    write_videos(data, output_dir, args.fps)

    # 4. Write episode/task meta
    print("\n[4/5] 写入 episode & task 元数据 ...")
    ep_rows = write_episodes_parquet(data, output_dir)
    write_tasks_parquet(output_dir, args.task)

    # 5. Write info.json
    print("\n[5/5] 写入 info.json ...")
    n_episodes = len(ep_rows)
    state_dim = data["states"].shape[1]
    img_h, img_w = data["images_up"].shape[1], data["images_up"].shape[2]
    write_info_json(output_dir, args.fps, n_frames, n_episodes,
                    args.task, args.repo_id, state_dim, img_h, img_w)

    # Summary
    print(f"\n{'='*60}")
    print(f"  转换完成!")
    print(f"{'='*60}")
    print(f"  总帧数:     {n_frames}")
    print(f"  Episode 数: {n_episodes}")
    print(f"  State dim:  {state_dim}")
    print(f"  Image:      {img_w}×{img_h}")
    print(f"  FPS:        {args.fps}")
    print(f"  输出目录:   {output_dir}")
    print()

    # List output tree
    print("  目录结构:")
    for p in sorted(output_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(output_dir)
            size_kb = p.stat().st_size / 1024
            print(f"    {rel}  ({size_kb:.1f} KB)")
    print()


if __name__ == "__main__":
    main()
