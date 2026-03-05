"""
SDG SO-101 改进版 — 修复朝向 / 自动探测 IK / .rrd 输出
=====================================================
基于 improved.md 的优化方案：
  1. URDF 加载位置修正（底座居中到世界原点）
  2. Probe 阶段：零位渲染 + 自动探测最佳 IK 目标四元数
  3. PD 增益提升 (2-5x)
  4. 相机位置调整（对准修正后的机器人）
  5. 内置 .rrd 输出（probe 诊断 + episode 数据）

用法：
  python sdg_so101_improved.py                           # 1 episode, 默认参数
  python sdg_so101_improved.py --euler 180 0 0           # 尝试 X 轴翻转
  python sdg_so101_improved.py --episodes 3 --save /out  # 多 episode
"""
import argparse
import os
import subprocess
import sys
import time
import urllib.request
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

parser = argparse.ArgumentParser(description="SDG SO-101 改进版 (probe + .rrd)")
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--episode-length", type=float, default=8.0)
parser.add_argument("--img-w", type=int, default=640)
parser.add_argument("--img-h", type=int, default=480)
parser.add_argument("--save", default="/output")
parser.add_argument("--urdf-dir", default=None)
parser.add_argument("--no-download", action="store_true")
parser.add_argument("--cpu", action="store_true")
parser.add_argument(
    "--euler", nargs=3, type=float, default=[0.0, 0.0, 0.0],
    help="URDF base euler correction in degrees (rx ry rz)",
)
args = parser.parse_args()

# ── Constants ────────────────────────────────────────────────────────────────

HF_REPO = "haixuantao/dora-bambot"
HF_BASE = f"https://huggingface.co/{HF_REPO}/resolve/main/URDF"
URDF_FILENAME = "so101.urdf"
STL_FILES = [
    "base_motor_holder_so101_v1.stl", "base_so101_v2.stl",
    "sts3215_03a_v1.stl", "waveshare_mounting_plate_so101_v2.stl",
    "motor_holder_so101_base_v1.stl", "rotation_pitch_so101_v1.stl",
    "upper_arm_so101_v1.stl", "under_arm_so101_v1.stl",
    "motor_holder_so101_wrist_v1.stl", "sts3215_03a_no_horn_v1.stl",
    "wrist_roll_pitch_so101_v2.stl", "wrist_roll_follower_so101_v1.stl",
    "moving_jaw_so101_v1.stl",
]

JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

# From URDF <limit> tags (radians)
JOINT_LIMITS_RAD = np.array([
    [-1.91986, 1.91986],   # 1: shoulder_pan
    [-1.74533, 1.74533],   # 2: shoulder_lift
    [-1.74533, 1.5708],    # 3: elbow_flex
    [-1.65806, 1.65806],   # 4: wrist_flex
    [-2.74385, 2.84121],   # 5: wrist_roll
    [-0.174533, 1.74533],  # 6: gripper
])

# URDF baseframe offset (from URDF fixed joint "baseframe_frame")
BASE_OFFSET = (0.163, 0.168, 0.0)

# Improved PD gains (2-5x original)
KP = np.array([500.0, 500.0, 400.0, 400.0, 300.0, 200.0])
KV = np.array([50.0,  50.0,  40.0,  40.0,  30.0,  20.0])

# ── Helpers ──────────────────────────────────────────────────────────────────

def stage(name):
    print(f"\n{'─'*60}\n  [{name}]\n{'─'*60}")

def to_numpy(t):
    arr = t.cpu().numpy() if hasattr(t, 'cpu') else np.array(t)
    return arr[0] if arr.ndim > 1 else arr

def render_camera(cam):
    rgb, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, normal=False)
    arr = rgb.cpu().numpy() if hasattr(rgb, 'cpu') else np.array(rgb)
    if arr.ndim == 4:
        arr = arr[0]
    return arr.astype(np.uint8)

def quat_to_rotmat(q):
    """Quaternion (w,x,y,z) → 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])

def joint_comfort_score(q_rad, limits):
    """Lower is better — measures how close joints are to their limits."""
    n = min(len(q_rad), len(limits))
    score = 0.0
    for i in range(n):
        lo, hi = limits[i]
        span = hi - lo
        if span < 1e-6:
            continue
        mid = (lo + hi) / 2.0
        normalized = (q_rad[i] - mid) / (span / 2.0)
        score += normalized ** 2
    return score

# ── [1] URDF Download ───────────────────────────────────────────────────────
stage("1/8  URDF 下载")

def download_urdf(target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = target_dir / "assets"
    assets_dir.mkdir(exist_ok=True)
    urdf_path = target_dir / URDF_FILENAME
    if not urdf_path.exists():
        url = f"{HF_BASE}/{URDF_FILENAME}"
        print(f"  downloading {URDF_FILENAME} ...")
        urllib.request.urlretrieve(url, str(urdf_path))
    else:
        print(f"  {URDF_FILENAME} exists")
    for stl in STL_FILES:
        stl_path = assets_dir / stl
        if not stl_path.exists():
            url = f"{HF_BASE}/assets/{stl}"
            print(f"  downloading assets/{stl} ...")
            urllib.request.urlretrieve(url, str(stl_path))
    return urdf_path

urdf_dir = Path(args.urdf_dir) if args.urdf_dir else Path(__file__).resolve().parent / "models" / "so101_urdf"
if args.no_download:
    urdf_path = urdf_dir / URDF_FILENAME
    assert urdf_path.exists(), f"URDF not found: {urdf_path}"
else:
    urdf_path = download_urdf(urdf_dir)
print(f"  ✓ URDF: {urdf_path}")

# ── [2] Genesis + Scene ─────────────────────────────────────────────────────
stage("2/8  Genesis 初始化 + 场景构建")

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
    gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=(0.05, 0.0, 0.015)),
    surface=gs.surfaces.Default(color=(1.0, 0.3, 0.3, 1.0)),
)

euler_deg = tuple(args.euler)
print(f"  URDF pos={BASE_OFFSET}, euler={euler_deg}")

so101 = scene.add_entity(
    gs.morphs.URDF(
        file=str(urdf_path),
        pos=BASE_OFFSET,
        euler=euler_deg,
        fixed=True,
    )
)

cam_up = scene.add_camera(
    res=(args.img_w, args.img_h),
    pos=(0.0, 0.0, 0.55),
    lookat=(0.0, 0.0, 0.08),
    fov=60,
    GUI=False,
)
cam_side = scene.add_camera(
    res=(args.img_w, args.img_h),
    pos=(0.35, -0.30, 0.20),
    lookat=(0.0, 0.0, 0.10),
    fov=55,
    GUI=False,
)
cam_front = scene.add_camera(
    res=(args.img_w, args.img_h),
    pos=(0.35, 0.0, 0.15),
    lookat=(0.0, 0.0, 0.10),
    fov=55,
    GUI=False,
)

scene.build()
print("  ✓ scene built")

# ── [3] Joint Discovery ─────────────────────────────────────────────────────
stage("3/8  关节发现")

n_dofs = so101.n_dofs
ALL_DOF_IDX = np.arange(n_dofs)
print(f"  n_dofs = {n_dofs}")

for j in so101.joints:
    print(f"    joint: {j.name}")

ee_link = None
for candidate in ["gripperframe", "gripper", "moving_jaw_so101_v1"]:
    try:
        ee_link = so101.get_link(candidate)
        print(f"  ✓ EE link = {candidate}")
        break
    except Exception:
        pass
if ee_link is None:
    print("  ✗ EE link not found — IK unavailable")
    sys.exit(1)

# ── [4] PD Gains ─────────────────────────────────────────────────────────────
stage("4/8  PD 增益")

kp = KP[:n_dofs]
kv = KV[:n_dofs]
so101.set_dofs_kp(kp, ALL_DOF_IDX)
so101.set_dofs_kv(kv, ALL_DOF_IDX)
print(f"  kp = {kp.tolist()}")
print(f"  kv = {kv.tolist()}")

# ── [5] Probe Phase ─────────────────────────────────────────────────────────
stage("5/8  Probe — 零位诊断 + IK 探测")

so101.set_qpos(np.zeros(n_dofs))
for _ in range(60):
    scene.step()

# Read all link world poses
print("\n  ── Link world poses (zero-config) ──")
for link in so101.links:
    pos = to_numpy(link.get_pos())
    print(f"    {link.name:30s}  pos=[{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]")

ee_pos_zero = to_numpy(ee_link.get_pos())
ee_quat_zero = to_numpy(ee_link.get_quat())
print(f"\n  EE zero-config:")
print(f"    pos  = {ee_pos_zero}")
print(f"    quat = {ee_quat_zero}")

R_ee = quat_to_rotmat(ee_quat_zero)
ee_z_axis = R_ee[:, 2]
print(f"    Z-axis in world = [{ee_z_axis[0]:+.3f}, {ee_z_axis[1]:+.3f}, {ee_z_axis[2]:+.3f}]")
if ee_z_axis[2] > 0.3:
    print("    → gripper Z points UP — likely inverted")
elif ee_z_axis[2] < -0.3:
    print("    → gripper Z points DOWN — correct orientation")
else:
    print("    → gripper Z is sideways — check orientation")

# Render probe views
probe_up = render_camera(cam_up)
probe_side = render_camera(cam_side)
probe_front = render_camera(cam_front)

# IK quaternion candidate search
print("\n  ── IK quaternion candidates ──")
IK_CANDIDATES = [
    ("identity",     np.array([1, 0, 0, 0], dtype=np.float64)),
    ("180x",         np.array([0, 1, 0, 0], dtype=np.float64)),
    ("180y",         np.array([0, 0, 1, 0], dtype=np.float64)),
    ("180z",         np.array([0, 0, 0, 1], dtype=np.float64)),
    ("90x",          np.array([0.7071, 0.7071, 0, 0], dtype=np.float64)),
    ("-90x",         np.array([0.7071, -0.7071, 0, 0], dtype=np.float64)),
    ("90y",          np.array([0.7071, 0, 0.7071, 0], dtype=np.float64)),
    ("-90y",         np.array([0.7071, 0, -0.7071, 0], dtype=np.float64)),
    ("90z",          np.array([0.7071, 0, 0, 0.7071], dtype=np.float64)),
    ("-90z",         np.array([0.7071, 0, 0, -0.7071], dtype=np.float64)),
    ("zero_config",  ee_quat_zero.copy()),
]

target_ik_pos = np.array([0.05, 0.0, 0.05])

best_name = None
best_quat = None
best_score = float('inf')
best_qdeg = None

limits = JOINT_LIMITS_RAD[:n_dofs]

for name, quat in IK_CANDIDATES:
    try:
        q = so101.inverse_kinematics(link=ee_link, pos=target_ik_pos, quat=quat)
        q_np = to_numpy(q)
        q_deg = np.rad2deg(q_np)

        within = True
        for i in range(min(n_dofs, len(limits))):
            if q_np[i] < limits[i, 0] - 0.01 or q_np[i] > limits[i, 1] + 0.01:
                within = False
                break

        score = joint_comfort_score(q_np, limits)
        status = "✓" if within else "✗ OOL"
        print(f"    {name:15s}  score={score:6.2f}  {status}  deg={np.round(q_deg, 1).tolist()}")

        if within and score < best_score:
            best_score = score
            best_quat = quat.copy()
            best_name = name
            best_qdeg = q_deg.copy()
    except Exception as e:
        print(f"    {name:15s}  FAIL: {e}")

if best_quat is None:
    print("\n  ⚠ No candidate within limits — falling back to 180x")
    best_quat = np.array([0, 1, 0, 0], dtype=np.float64)
    best_name = "180x (fallback)"
    best_qdeg = None

print(f"\n  ✓ Best IK quat: {best_name}  q={best_quat.tolist()}")
if best_qdeg is not None:
    print(f"    → IK result (deg): {np.round(best_qdeg, 1).tolist()}")

# Determine HOME: use IK result at a "rest" position above origin
print("\n  ── HOME pose determination ──")
try:
    rest_pos = np.array([0.0, 0.0, 0.15])
    q_home = so101.inverse_kinematics(link=ee_link, pos=rest_pos, quat=best_quat)
    HOME_RAD = to_numpy(q_home)
    HOME_DEG = np.rad2deg(HOME_RAD)
    home_within = all(
        limits[i, 0] - 0.01 <= HOME_RAD[i] <= limits[i, 1] + 0.01
        for i in range(min(n_dofs, len(limits)))
    )
    if home_within:
        print(f"  ✓ HOME (IK-derived): {np.round(HOME_DEG, 1).tolist()}")
    else:
        print(f"  ⚠ HOME IK out of limits, using zeros")
        HOME_RAD = np.zeros(n_dofs)
        HOME_DEG = np.zeros(n_dofs)
except Exception:
    print(f"  ⚠ HOME IK failed, using zeros")
    HOME_RAD = np.zeros(n_dofs)
    HOME_DEG = np.zeros(n_dofs)

# Apply HOME and render
so101.set_qpos(HOME_RAD)
so101.control_dofs_position(HOME_RAD, ALL_DOF_IDX)
for _ in range(80):
    scene.step()

cur_deg = np.rad2deg(to_numpy(so101.get_dofs_position(ALL_DOF_IDX)))
home_err = np.abs(cur_deg - HOME_DEG).mean()
print(f"  HOME tracking: mean_err={home_err:.2f}°, actual={np.round(cur_deg, 1).tolist()}")

probe_home_up = render_camera(cam_up)
probe_home_side = render_camera(cam_side)
probe_home_front = render_camera(cam_front)

# ── [6] Data Collection ─────────────────────────────────────────────────────
stage(f"6/8  数据采集 × {args.episodes} episodes")

steps_per_episode = int(args.episode_length * args.fps)

def generate_trajectory(n_steps, cube_pos):
    """IK-based pick-place trajectory with the probed quaternion."""
    traj = []

    def solve_ik(pos, gripper_deg=0.0):
        q = so101.inverse_kinematics(link=ee_link, pos=pos, quat=best_quat)
        q_np = to_numpy(q)
        q_deg = np.rad2deg(q_np)
        if n_dofs >= 6:
            q_deg[5] = gripper_deg
        return q_deg

    def lerp(a, b, steps):
        return [a + (b - a) * (i + 1) / steps for i in range(steps)]

    try:
        pre_pos = cube_pos + np.array([0.0, 0.0, 0.10])
        q_pre = solve_ik(pre_pos, 0.0)

        approach_pos = cube_pos + np.array([0.0, 0.0, 0.02])
        q_approach = solve_ik(approach_pos, 0.0)

        lift_pos = cube_pos + np.array([0.0, 0.0, 0.15])
        q_lift = solve_ik(lift_pos, 25.0)

        ps = n_steps // 6
        rem = n_steps - ps * 6

        traj += lerp(HOME_DEG, q_pre, ps)         # home → pre-grasp
        traj += lerp(q_pre, q_approach, ps)        # lower to cube
        q_close = q_approach.copy()
        if n_dofs >= 6:
            q_close[5] = 25.0
        traj += lerp(q_approach, q_close, ps)      # close gripper
        traj += lerp(q_close, q_lift, ps)           # lift
        traj += [q_lift.copy()] * ps                # hold
        traj += lerp(q_lift, HOME_DEG, ps + rem)   # return home
        return traj
    except Exception as e:
        print(f"  ⚠ IK trajectory failed ({e}), using sinusoidal fallback")

    for i in range(n_steps):
        t = i / max(n_steps - 1, 1)
        phase = np.sin(np.pi * t)
        offset = np.zeros(n_dofs)
        for j in range(min(n_dofs, 5)):
            offset[j] = 10.0 * phase * np.sin(2 * np.pi * (j + 1) * t)
        traj.append(HOME_DEG + offset)
    return traj

all_episodes = []
t0_collect = time.time()

for ep in range(args.episodes):
    ep_t0 = time.time()

    so101.set_qpos(HOME_RAD)
    so101.control_dofs_position(HOME_RAD, ALL_DOF_IDX)
    for _ in range(40):
        scene.step()

    # Domain randomization: cube position
    cx = np.random.uniform(0.02, 0.10)
    cy = np.random.uniform(-0.06, 0.06)
    cz = 0.015
    cube_pos_rand = np.array([cx, cy, cz])
    cube.set_pos(
        torch.tensor(cube_pos_rand, dtype=torch.float32, device=gs.device).unsqueeze(0)
    )
    for _ in range(15):
        scene.step()
    cube_pos_actual = to_numpy(cube.get_pos())

    trajectory = generate_trajectory(steps_per_episode, cube_pos_actual)

    ep_data = {
        "observation.state": [], "action": [],
        "observation.images.up": [], "observation.images.side": [],
        "timestamp": [], "frame_index": [], "episode_index": [],
    }

    for fi, target_deg in enumerate(trajectory):
        target_rad = np.deg2rad(np.array(target_deg, dtype=np.float32))
        so101.control_dofs_position(target_rad, ALL_DOF_IDX)
        scene.step()

        state_deg = np.rad2deg(to_numpy(so101.get_dofs_position(ALL_DOF_IDX)))
        ep_data["observation.state"].append(state_deg.astype(np.float32))
        ep_data["action"].append(np.array(target_deg, dtype=np.float32))
        ep_data["observation.images.up"].append(render_camera(cam_up))
        ep_data["observation.images.side"].append(render_camera(cam_side))
        ep_data["timestamp"].append(fi / args.fps)
        ep_data["frame_index"].append(fi)
        ep_data["episode_index"].append(ep)

    all_episodes.append(ep_data)
    n_frames = len(trajectory)
    elapsed = time.time() - ep_t0
    print(f"  episode {ep}: {n_frames} frames, {elapsed:.1f}s ({n_frames/elapsed:.0f} sim-fps)")

total_frames = sum(len(e["timestamp"]) for e in all_episodes)
print(f"  ✓ {args.episodes} episodes, {total_frames} frames, {time.time()-t0_collect:.1f}s")

# ── [7] Save npy ─────────────────────────────────────────────────────────────
stage("7/8  保存 npy")

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

# ── [8] Save .rrd ────────────────────────────────────────────────────────────
stage("8/8  生成 .rrd")

try:
    import rerun as rr

    rrd_path = out_dir / "improved_sdg.rrd"
    rr.init("so101_sdg_improved", spawn=False)

    # Probe section: zero-config views (frame -3..-1)
    rr.set_time("frame_index", sequence=-3)
    rr.set_time("timestamp", timestamp=-0.1)
    rr.log("probe/zero_config/cam_up", rr.Image(probe_up))
    rr.log("probe/zero_config/cam_side", rr.Image(probe_side))
    rr.log("probe/zero_config/cam_front", rr.Image(probe_front))

    rr.set_time("frame_index", sequence=-1)
    rr.set_time("timestamp", timestamp=-0.03)
    rr.log("probe/home_pose/cam_up", rr.Image(probe_home_up))
    rr.log("probe/home_pose/cam_side", rr.Image(probe_home_side))
    rr.log("probe/home_pose/cam_front", rr.Image(probe_home_front))

    # Episode data
    n_joints = min(all_states.shape[1], len(JOINT_NAMES))
    for i in range(len(all_states)):
        fi = int(all_fi[i])
        t = float(all_ts[i])
        rr.set_time("frame_index", sequence=fi)
        rr.set_time("timestamp", timestamp=t)

        rr.log("observation.images.up", rr.Image(all_imgs_up[i]))
        rr.log("observation.images.side", rr.Image(all_imgs_side[i]))

        for j in range(n_joints):
            jn = JOINT_NAMES[j]
            rr.log(f"state/{jn}", rr.Scalars(float(all_states[i, j])))
            rr.log(f"action/{jn}", rr.Scalars(float(all_actions[i, j])))
            rr.log(f"tracking/{jn}/state", rr.Scalars(float(all_states[i, j])))
            rr.log(f"tracking/{jn}/action", rr.Scalars(float(all_actions[i, j])))

    rr.save(str(rrd_path))
    size_mb = rrd_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ {rrd_path} ({size_mb:.1f} MB)")
except ImportError:
    print("  ⚠ rerun not installed — skipping .rrd")
except Exception as e:
    print(f"  ⚠ .rrd generation failed: {e}")

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'═'*60}")
print("  SUMMARY — SDG SO-101 Improved")
print(f"{'═'*60}")
print(f"  URDF pos   = {BASE_OFFSET}")
print(f"  URDF euler = {euler_deg}")
print(f"  IK quat    = {best_name} {best_quat.tolist()}")
print(f"  HOME (deg) = {np.round(HOME_DEG, 1).tolist()}")
print(f"  PD gains   = kp={kp.tolist()}, kv={kv.tolist()}")
print(f"  Episodes   = {args.episodes}, frames = {total_frames}")
print(f"  State  range: [{all_states.min():.1f}°, {all_states.max():.1f}°]")
print(f"  Action range: [{all_actions.min():.1f}°, {all_actions.max():.1f}°]")
print(f"  Output: {out_dir}")
print(f"{'═'*60}\n")
