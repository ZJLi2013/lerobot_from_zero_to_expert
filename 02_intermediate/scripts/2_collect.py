"""
SDG SO-101 采集脚本 — MJCF + 双相机 + .rrd 输出
================================================
使用 LeRobot 官方 SO-101 MJCF (so101_new_calib.xml) 加载机械臂，
执行 scripted pick-place 轨迹，输出 npy + .rrd。

用法：
  python 2_collect.py                           # 1 episode, 默认参数
  python 2_collect.py --episodes 3 --save /out  # 多 episode
  python 2_collect.py --xml /path/to/so101.xml  # 指定 XML 路径
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


# ── Headless display ─────────────────────────────────────────────────────────

def ensure_display():
    if os.environ.get("DISPLAY"):
        print(f"[display] DISPLAY={os.environ['DISPLAY']}")
        return
    xvfb = subprocess.run(["which", "Xvfb"], capture_output=True)
    if xvfb.returncode != 0:
        print("[display] WARNING: Xvfb not found")
        return
    print("[display] Starting Xvfb :99 ...")
    proc = subprocess.Popen(
        ["Xvfb", ":99", "-screen", "0", "1280x1024x24", "-ac", "+extension", "GLX"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    os.environ["DISPLAY"] = ":99"
    time.sleep(2)
    if proc.poll() is None:
        print(f"[display] Xvfb started (PID={proc.pid})")
    else:
        print("[display] WARNING: Xvfb exited immediately")


ensure_display()

# ── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="SDG SO-101 采集 (MJCF + .rrd)")
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--episode-length", type=float, default=8.0)
parser.add_argument("--img-w", type=int, default=640)
parser.add_argument("--img-h", type=int, default=480)
parser.add_argument("--save", default="/output")
parser.add_argument("--xml", default=None, help="Path to so101_new_calib.xml (auto-detect if omitted)")
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()

# ── Constants ────────────────────────────────────────────────────────────────

JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

HOME_DEG = np.array([0.0, -30.0, 90.0, -60.0, 0.0, 0.0], dtype=np.float32)
IK_QUAT_DOWN = np.array([0.0, 1.0, 0.0, 0.0])  # gripper pointing down

KP = np.array([500.0, 500.0, 400.0, 400.0, 300.0, 200.0])
KV = np.array([50.0, 50.0, 40.0, 40.0, 30.0, 20.0])

# ── Helpers ──────────────────────────────────────────────────────────────────

def stage(name):
    print(f"\n{'─'*60}\n  [{name}]\n{'─'*60}")


def to_numpy(t):
    arr = t.cpu().numpy() if hasattr(t, "cpu") else np.array(t)
    return arr[0] if arr.ndim > 1 else arr


def render_camera(cam):
    rgb, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, normal=False)
    arr = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.array(rgb)
    if arr.ndim == 4:
        arr = arr[0]
    return arr.astype(np.uint8)


def find_so101_xml(user_path=None):
    """Search for so101_new_calib.xml in common locations, auto-download from HF if needed."""
    if user_path:
        p = Path(user_path)
        if p.exists():
            return p
        return None

    candidates = [
        Path("assets/so101_new_calib.xml"),
        Path("02_intermediate/scripts/assets/so101_new_calib.xml"),
    ]
    try:
        import lerobot
        base = Path(lerobot.__file__).parent
        for sub in [
            "common/robot_devices/robots/assets",
            "common/robot_devices/motors/assets",
            "configs/robot",
        ]:
            p = base / sub / "so101_new_calib.xml"
            if p.exists():
                candidates.insert(0, p)
    except ImportError:
        pass

    for p in candidates:
        if p.exists():
            return p

    try:
        from huggingface_hub import snapshot_download
        print("  … downloading SO101 MJCF from Genesis-Intelligence/assets")
        asset_dir = snapshot_download(
            repo_type="dataset",
            repo_id="Genesis-Intelligence/assets",
            allow_patterns="SO101/*",
            max_workers=1,
        )
        p = Path(asset_dir) / "SO101" / "so101_new_calib.xml"
        if p.exists():
            return p
    except Exception as e:
        print(f"  … HF download failed: {e}")

    return None


# ── [1] Locate MJCF ─────────────────────────────────────────────────────────
stage("1/7  定位 SO-101 MJCF")

xml_path = find_so101_xml(args.xml)
if xml_path is None:
    print("  ✗ so101_new_calib.xml 未找到")
    print("    请通过 --xml 指定路径，或确认 lerobot 已安装")
    print("    (pip install lerobot)")
    sys.exit(1)

print(f"  ✓ {xml_path}")

# ── [2] Genesis + Scene ──────────────────────────────────────────────────────
stage("2/7  Genesis 初始化 + 场景构建")

import torch
import genesis as gs

backend = gs.cpu if args.cpu else gs.gpu
gs.init(backend=backend, logging_level="warning")
print(f"  backend={'cpu' if args.cpu else 'gpu'}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=1.0 / args.fps, substeps=4),
    rigid_options=gs.options.RigidOptions(
        enable_collision=True,
        enable_joint_limit=True,
    ),
    show_viewer=False,
)

scene.add_entity(gs.morphs.Plane())

cube = scene.add_entity(
    gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=(0.15, 0.0, 0.015)),
    surface=gs.surfaces.Default(color=(1.0, 0.3, 0.3, 1.0)),
)

so101 = scene.add_entity(
    gs.morphs.MJCF(file=str(xml_path), pos=(0.0, 0.0, 0.0))
)
print(f"  ✓ SO-101 loaded via MJCF")

cam_up = scene.add_camera(
    res=(args.img_w, args.img_h),
    pos=(0.0, 0.0, 0.7),
    lookat=(0.15, 0.0, 0.0),
    fov=55, GUI=False,
)
cam_side = scene.add_camera(
    res=(args.img_w, args.img_h),
    pos=(0.5, -0.4, 0.3),
    lookat=(0.15, 0.0, 0.1),
    fov=45, GUI=False,
)

scene.build()
print("  ✓ scene built")

# ── [3] Joint Discovery ──────────────────────────────────────────────────────
stage("3/7  关节发现")

n_dofs = so101.n_dofs
ALL_DOF_IDX = np.arange(n_dofs)
print(f"  n_dofs = {n_dofs}")

for j in so101.joints:
    print(f"    joint: {j.name}")

ee_link = None
for candidate in ["gripper_link", "gripperframe", "gripper", "Fixed_Jaw"]:
    try:
        ee_link = so101.get_link(candidate)
        print(f"  ✓ EE link = {candidate}")
        break
    except Exception:
        pass
if ee_link is None:
    print("  ✗ EE link not found")
    sys.exit(1)

# ── [4] PD Gains + Home ─────────────────────────────────────────────────────
stage("4/7  PD 增益 + Home 姿态")

kp = KP[:n_dofs]
kv = KV[:n_dofs]
so101.set_dofs_kp(kp, ALL_DOF_IDX)
so101.set_dofs_kv(kv, ALL_DOF_IDX)
print(f"  kp = {kp.tolist()}")
print(f"  kv = {kv.tolist()}")

home_deg = HOME_DEG[:n_dofs]
home_rad = np.deg2rad(home_deg)
so101.set_qpos(home_rad)
so101.control_dofs_position(home_rad, ALL_DOF_IDX)
for _ in range(60):
    scene.step()

cur_deg = np.rad2deg(to_numpy(so101.get_dofs_position(ALL_DOF_IDX)))
home_err = np.abs(cur_deg - home_deg).mean()
print(f"  HOME = {home_deg.tolist()}")
print(f"  tracking: mean_err={home_err:.2f}°, actual={np.round(cur_deg, 1).tolist()}")

# Quick IK sanity check
try:
    test_pos = np.array([0.15, 0.0, 0.08])
    q_test = so101.inverse_kinematics(link=ee_link, pos=test_pos, quat=IK_QUAT_DOWN)
    q_deg = np.rad2deg(to_numpy(q_test))
    print(f"  ✓ IK sanity check: target={test_pos.tolist()} → {np.round(q_deg, 1).tolist()}")
except Exception as e:
    print(f"  ✗ IK failed: {e}")
    sys.exit(1)

# ── [5] Data Collection ──────────────────────────────────────────────────────
stage(f"5/7  数据采集 × {args.episodes} episodes")

steps_per_episode = int(args.episode_length * args.fps)


def generate_trajectory(n_steps, cube_pos):
    """IK-based pick-place trajectory."""
    traj = []

    def solve_ik(pos, gripper_deg=0.0):
        q = so101.inverse_kinematics(link=ee_link, pos=pos, quat=IK_QUAT_DOWN)
        q_np = to_numpy(q)
        q_deg = np.rad2deg(q_np)
        if n_dofs >= 6:
            q_deg[5] = gripper_deg
        return q_deg

    def lerp(a, b, steps):
        return [a + (b - a) * (i + 1) / steps for i in range(steps)]

    pre_pos = cube_pos + np.array([0.0, 0.0, 0.10])
    q_pre = solve_ik(pre_pos, 0.0)

    approach_pos = cube_pos + np.array([0.0, 0.0, 0.02])
    q_approach = solve_ik(approach_pos, 0.0)

    lift_pos = cube_pos + np.array([0.0, 0.0, 0.15])
    q_lift = solve_ik(lift_pos, 25.0)

    ps = n_steps // 6
    rem = n_steps - ps * 6

    traj += lerp(home_deg, q_pre, ps)
    traj += lerp(q_pre, q_approach, ps)
    q_close = q_approach.copy()
    if n_dofs >= 6:
        q_close[5] = 25.0
    traj += lerp(q_approach, q_close, ps)
    traj += lerp(q_close, q_lift, ps)
    traj += [q_lift.copy()] * ps
    traj += lerp(q_lift, home_deg, ps + rem)
    return traj


all_episodes = []
t0_collect = time.time()

for ep in range(args.episodes):
    ep_t0 = time.time()

    so101.set_qpos(home_rad)
    so101.control_dofs_position(home_rad, ALL_DOF_IDX)
    for _ in range(40):
        scene.step()

    cx = np.random.uniform(0.10, 0.22)
    cy = np.random.uniform(-0.08, 0.08)
    cube_pos_rand = np.array([cx, cy, 0.015])
    cube.set_pos(
        torch.tensor(cube_pos_rand, dtype=torch.float32, device=gs.device).unsqueeze(0)
    )
    for _ in range(15):
        scene.step()
    cube_pos_actual = to_numpy(cube.get_pos())

    try:
        trajectory = generate_trajectory(steps_per_episode, cube_pos_actual)
    except Exception as e:
        print(f"  ⚠ episode {ep}: IK failed ({e}), skipping")
        continue

    ps = steps_per_episode // 6
    close_start_fi = 2 * ps
    lift_end_fi = 4 * ps

    ep_data = {
        "observation.state": [],
        "action": [],
        "observation.images.up": [],
        "observation.images.side": [],
        "timestamp": [],
        "frame_index": [],
        "episode_index": [],
    }

    cube_z_before = None
    cube_z_after = None

    for fi, target_deg in enumerate(trajectory):
        target_rad = np.deg2rad(np.array(target_deg, dtype=np.float32))
        so101.control_dofs_position(target_rad, ALL_DOF_IDX)
        scene.step()

        if fi == close_start_fi:
            cube_z_before = float(to_numpy(cube.get_pos())[2])
        if fi == lift_end_fi:
            cube_z_after = float(to_numpy(cube.get_pos())[2])

        state_deg = np.rad2deg(to_numpy(so101.get_dofs_position(ALL_DOF_IDX)))
        ep_data["observation.state"].append(state_deg.astype(np.float32))
        ep_data["action"].append(np.array(target_deg, dtype=np.float32))
        ep_data["observation.images.up"].append(render_camera(cam_up))
        ep_data["observation.images.side"].append(render_camera(cam_side))
        ep_data["timestamp"].append(fi / args.fps)
        ep_data["frame_index"].append(fi)
        ep_data["episode_index"].append(ep)

    if cube_z_before is None:
        cube_z_before = float(to_numpy(cube.get_pos())[2])
    if cube_z_after is None:
        cube_z_after = float(to_numpy(cube.get_pos())[2])
    lift_delta = cube_z_after - cube_z_before
    grasp_ok = lift_delta > 0.01

    all_episodes.append(ep_data)
    n_frames = len(trajectory)
    elapsed = time.time() - ep_t0
    print(
        f"  episode {ep}: {n_frames} frames, {elapsed:.1f}s ({n_frames/elapsed:.0f} sim-fps) | "
        f"grasp={'✓' if grasp_ok else '✗'} delta_z={lift_delta:.4f}m "
        f"(before={cube_z_before:.4f}, after={cube_z_after:.4f})"
    )

if not all_episodes:
    print("  ✗ No episodes collected")
    sys.exit(1)

total_frames = sum(len(e["timestamp"]) for e in all_episodes)
print(f"  ✓ {len(all_episodes)} episodes, {total_frames} frames, {time.time()-t0_collect:.1f}s")

# ── [6] Save npy ─────────────────────────────────────────────────────────────
stage("6/7  保存 npy")

out_dir = Path(args.save)
out_dir.mkdir(parents=True, exist_ok=True)

all_states = np.concatenate([np.stack(e["observation.state"]) for e in all_episodes])
all_actions = np.concatenate([np.stack(e["action"]) for e in all_episodes])
all_imgs_up = np.concatenate([np.stack(e["observation.images.up"]) for e in all_episodes])
all_imgs_side = np.concatenate([np.stack(e["observation.images.side"]) for e in all_episodes])
all_ts = np.concatenate([np.array(e["timestamp"]) for e in all_episodes])
all_fi = np.concatenate([np.array(e["frame_index"]) for e in all_episodes])
all_ei = np.concatenate([np.array(e["episode_index"]) for e in all_episodes])

for name, arr in [
    ("states", all_states), ("actions", all_actions),
    ("images_up", all_imgs_up), ("images_side", all_imgs_side),
    ("timestamps", all_ts), ("frame_indices", all_fi),
    ("episode_indices", all_ei),
]:
    np.save(out_dir / f"{name}.npy", arr)
    print(f"  {name}: {arr.shape}")

print(f"  state  range: [{all_states.min():.1f}, {all_states.max():.1f}]°")
print(f"  action range: [{all_actions.min():.1f}, {all_actions.max():.1f}]°")

# ── [7] Save .rrd ────────────────────────────────────────────────────────────
stage("7/7  生成 .rrd")

try:
    import rerun as rr

    rrd_path = out_dir / "sdg_collect.rrd"
    rr.init("so101_sdg_collect", spawn=False)

    n_joints = min(all_states.shape[1], len(JOINT_NAMES))
    for i in range(len(all_states)):
        fi_val = int(all_fi[i])
        t = float(all_ts[i])
        rr.set_time("frame_index", sequence=fi_val)
        rr.set_time("timestamp", timestamp=t)

        rr.log("observation.images.up", rr.Image(all_imgs_up[i]))
        rr.log("observation.images.side", rr.Image(all_imgs_side[i]))

        for j in range(n_joints):
            jn = JOINT_NAMES[j]
            rr.log(f"state/{jn}", rr.Scalars(float(all_states[i, j])))
            rr.log(f"action/{jn}", rr.Scalars(float(all_actions[i, j])))

    rr.save(str(rrd_path))
    size_mb = rrd_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ {rrd_path} ({size_mb:.1f} MB)")
except ImportError:
    print("  ⚠ rerun not installed — skipping .rrd")
except Exception as e:
    print(f"  ⚠ .rrd generation failed: {e}")

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'═'*60}")
print("  SUMMARY — SDG SO-101 Collect")
print(f"{'═'*60}")
print(f"  MJCF     = {xml_path}")
print(f"  HOME°    = {home_deg.tolist()}")
print(f"  IK quat  = {IK_QUAT_DOWN.tolist()} (gripper-down)")
print(f"  PD gains = kp={kp.tolist()}, kv={kv.tolist()}")
print(f"  Episodes = {len(all_episodes)}, frames = {total_frames}")
print(f"  State  range: [{all_states.min():.1f}°, {all_states.max():.1f}°]")
print(f"  Action range: [{all_actions.min():.1f}°, {all_actions.max():.1f}°]")
print(f"  Output: {out_dir}")
print(f"{'═'*60}\n")
