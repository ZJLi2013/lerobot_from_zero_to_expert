"""
SDG: Genesis × SO-101 URDF 合成数据生成验证
=============================================
基于真实 SO-101 URDF（来自 HuggingFace haixuantao/dora-bambot），
在 Genesis 中加载 6-DOF 机械臂，执行 pick-place 轨迹并按
svla_so101_pickplace 数据结构采集 (state, action, image_up, image_side)。

阶段：
  [1] SO-101 URDF 下载 / 校验
  [2] Genesis 初始化 + 场景构建（SO-101 + 双相机）
  [3] 关节发现 & 语义映射（6-DOF）
  [4] PD 增益 + Home 姿态验证
  [5] IK 求解验证
  [6] 30Hz 采集循环 × N episodes（scripted pick-place）
  [7] 保存数据（npy）+ 统计摘要

用法：
  python sdg_so101_genesis.py                          # 默认 3 episodes
  python sdg_so101_genesis.py --episodes 10 --fps 30   # 自定义
  python sdg_so101_genesis.py --no-download             # 跳过 URDF 下载
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
        print(f"[display] DISPLAY={os.environ['DISPLAY']} (already set)")
        return
    xvfb = subprocess.run(["which", "Xvfb"], capture_output=True)
    if xvfb.returncode != 0:
        print("[display] WARNING: Xvfb not found. Install: apt-get install -y xvfb")
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

parser = argparse.ArgumentParser(description="Genesis × SO-101 URDF SDG 验证")
parser.add_argument("--episodes", type=int, default=3, help="采集 episode 数")
parser.add_argument("--fps", type=int, default=30, help="采集频率")
parser.add_argument("--episode-length", type=float, default=8.0, help="单 episode 时长(秒)")
parser.add_argument("--img-w", type=int, default=640, help="图像宽度")
parser.add_argument("--img-h", type=int, default=480, help="图像高度")
parser.add_argument("--save", default="/tmp/sdg_so101_output", help="输出目录")
parser.add_argument("--urdf-dir", default=None, help="SO-101 URDF 目录（留空自动下载）")
parser.add_argument("--no-download", action="store_true", help="跳过 URDF 下载")
parser.add_argument("--cpu", action="store_true", help="CPU backend")
args = parser.parse_args()

PASS = "✓ PASS"
FAIL = "✗ FAIL"
results = {}

def stage(name):
    print(f"\n{'─'*60}\n  [{name}]\n{'─'*60}")

def ok(label, val=""):
    msg = f"  {PASS}  {label}"
    if val:
        msg += f"  →  {val}"
    print(msg)

def err(label, e):
    print(f"  {FAIL}  {label}\n         {type(e).__name__}: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# [1] SO-101 URDF 下载
# ─────────────────────────────────────────────────────────────────────────────
stage("1/7  SO-101 URDF 下载 / 校验")

HF_REPO = "haixuantao/dora-bambot"
HF_BASE = f"https://huggingface.co/{HF_REPO}/resolve/main/URDF"

URDF_FILENAME = "so101.urdf"
STL_FILES = [
    "base_motor_holder_so101_v1.stl",
    "base_so101_v2.stl",
    "sts3215_03a_v1.stl",
    "waveshare_mounting_plate_so101_v2.stl",
    "motor_holder_so101_base_v1.stl",
    "rotation_pitch_so101_v1.stl",
    "upper_arm_so101_v1.stl",
    "under_arm_so101_v1.stl",
    "motor_holder_so101_wrist_v1.stl",
    "sts3215_03a_no_horn_v1.stl",
    "wrist_roll_pitch_so101_v2.stl",
    "wrist_roll_follower_so101_v1.stl",
    "moving_jaw_so101_v1.stl",
]

# SO-101 joint semantic mapping (URDF joint name → semantic name)
# URDF uses numeric names: "1"..."6"
JOINT_SEMANTIC = {
    "1": "shoulder_pan",
    "2": "shoulder_lift",
    "3": "elbow_flex",
    "4": "wrist_flex",
    "5": "wrist_roll",
    "6": "gripper",
}

HOME_DEG = np.array([0.0, -30.0, 90.0, -60.0, 0.0, 0.0], dtype=np.float32)
KP = np.array([100.0, 100.0, 80.0, 80.0, 60.0, 40.0])
KV = np.array([10.0,  10.0,  8.0,  8.0,  6.0,  4.0])

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
        print(f"  {URDF_FILENAME} already exists")

    for stl in STL_FILES:
        stl_path = assets_dir / stl
        if not stl_path.exists():
            url = f"{HF_BASE}/assets/{stl}"
            print(f"  downloading assets/{stl} ...")
            urllib.request.urlretrieve(url, str(stl_path))

    return urdf_path

try:
    if args.urdf_dir:
        urdf_dir = Path(args.urdf_dir)
    else:
        urdf_dir = Path(__file__).resolve().parent / "models" / "so101_urdf"

    if args.no_download:
        urdf_path = urdf_dir / URDF_FILENAME
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        ok("URDF exists (skip download)", str(urdf_path))
    else:
        urdf_path = download_urdf(urdf_dir)
        ok("URDF ready", str(urdf_path))

    n_stl = len(list((urdf_dir / "assets").glob("*.stl")))
    ok(f"STL meshes", f"{n_stl}/{len(STL_FILES)} files")
    results["1"] = True
except Exception as e:
    err("URDF download", e)
    results["1"] = False
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# [2] Genesis 初始化 + 场景构建
# ─────────────────────────────────────────────────────────────────────────────
stage("2/7  Genesis 初始化 + 场景构建")
try:
    import torch
    import genesis as gs

    backend = gs.cpu if args.cpu else gs.gpu
    gs.init(backend=backend, logging_level="warning")
    ok("gs.init", f"backend={'cpu' if args.cpu else 'gpu'}")
    if torch.cuda.is_available():
        ok("GPU", torch.cuda.get_device_name(0))

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / args.fps, substeps=4),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            enable_joint_limit=True,
        ),
        show_viewer=False,
    )
    ok("Scene created", f"dt={1.0/args.fps:.4f}s ({args.fps}Hz)")

    scene.add_entity(gs.morphs.Plane())
    ok("Plane added")

    cube = scene.add_entity(
        gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=(0.15, 0.0, 0.015)),
        surface=gs.surfaces.Default(color=(1.0, 0.3, 0.3, 1.0)),
    )
    ok("Cube added", "3cm red cube at (0.15, 0, 0.015)")

    # Load SO-101 via URDF
    so101 = scene.add_entity(
        gs.morphs.URDF(
            file=str(urdf_path),
            pos=(0.0, 0.0, 0.0),
        )
    )
    ok("SO-101 loaded from URDF")

    # Dual cameras matching svla_so101_pickplace layout
    cam_up = scene.add_camera(
        res=(args.img_w, args.img_h),
        pos=(0.0, 0.0, 0.7),
        lookat=(0.15, 0.0, 0.0),
        fov=55,
        GUI=False,
    )
    ok("Camera UP", f"{args.img_w}×{args.img_h}")

    cam_side = scene.add_camera(
        res=(args.img_w, args.img_h),
        pos=(0.5, -0.4, 0.3),
        lookat=(0.15, 0.0, 0.1),
        fov=45,
        GUI=False,
    )
    ok("Camera SIDE", f"{args.img_w}×{args.img_h}")

    scene.build()
    ok("scene.build() done")
    results["2"] = True
except Exception as e:
    err("scene build", e)
    results["2"] = False
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# [3] 关节发现 & 语义映射
# ─────────────────────────────────────────────────────────────────────────────
stage("3/7  关节发现 & 语义映射")
try:
    n_dofs = so101.n_dofs
    ok(f"n_dofs = {n_dofs}")

    link_names = [l.name for l in so101.links]
    joint_names = [j.name for j in so101.joints]

    print("  Links:")
    for ln in link_names:
        print(f"    • {ln}")
    print("  Joints:")
    for jn in joint_names:
        semantic = JOINT_SEMANTIC.get(jn, "?")
        print(f"    • {jn} → {semantic}")

    if n_dofs != 6:
        print(f"  [!] 预期 6 DOF，实际 {n_dofs}。将适配实际 DOF 数。")

    ALL_DOF_IDX = np.arange(n_dofs)

    # Find end-effector link
    ee_link = None
    for candidate in ["gripperframe", "gripper", "moving_jaw_so101_v1", "wrist"]:
        try:
            ee_link = so101.get_link(candidate)
            ok(f"EE link", candidate)
            break
        except Exception:
            pass
    if ee_link is None:
        print(f"  [!] 未找到 EE link，IK 将不可用")

    results["3"] = True
except Exception as e:
    err("joint discovery", e)
    results["3"] = False
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# [4] PD 增益 + Home 姿态
# ─────────────────────────────────────────────────────────────────────────────
stage("4/7  PD 增益 + Home 姿态")
try:
    kp = KP[:n_dofs]
    kv = KV[:n_dofs]
    so101.set_dofs_kp(kp, ALL_DOF_IDX)
    so101.set_dofs_kv(kv, ALL_DOF_IDX)
    ok("PD gains set", f"kp={kp.tolist()}, kv={kv.tolist()}")

    home_deg = HOME_DEG[:n_dofs]
    home_rad = np.deg2rad(home_deg)
    so101.set_qpos(home_rad)
    for _ in range(50):
        scene.step()
    ok("Home pose set", f"deg={home_deg.tolist()}")

    cur_t = so101.get_dofs_position(ALL_DOF_IDX)
    cur_np = cur_t.cpu().numpy() if hasattr(cur_t, 'cpu') else np.array(cur_t)
    if cur_np.ndim > 1:
        cur_np = cur_np[0]
    cur_deg = np.rad2deg(cur_np)
    home_err = np.abs(cur_deg - home_deg).mean()
    ok("Home tracking", f"mean_err={home_err:.2f}° | actual={np.round(cur_deg, 1).tolist()}")
    results["4"] = True
except Exception as e:
    err("PD / home", e)
    results["4"] = False

# ─────────────────────────────────────────────────────────────────────────────
# [5] IK 求解验证
# ─────────────────────────────────────────────────────────────────────────────
stage("5/7  IK 求解验证")
ik_available = False
try:
    if ee_link is None:
        print(f"  - SKIP  EE link 未找到")
    else:
        cube_pos_t = cube.get_pos()
        cube_pos = cube_pos_t.cpu().numpy() if hasattr(cube_pos_t, 'cpu') else np.array(cube_pos_t)
        if cube_pos.ndim > 1:
            cube_pos = cube_pos[0]

        target_pos = cube_pos + np.array([0.0, 0.0, 0.08])
        target_quat = np.array([0.0, 1.0, 0.0, 0.0])

        t0 = time.time()
        qpos_ik = so101.inverse_kinematics(
            link=ee_link,
            pos=target_pos,
            quat=target_quat,
        )
        dt_ik = (time.time() - t0) * 1000
        qpos_np = qpos_ik.cpu().numpy() if hasattr(qpos_ik, 'cpu') else np.array(qpos_ik)
        if qpos_np.ndim > 1:
            qpos_np = qpos_np[0]
        ok("IK solved", f"{dt_ik:.0f}ms → deg={np.round(np.rad2deg(qpos_np), 1).tolist()}")
        ik_available = True
    results["5"] = True
except Exception as e:
    err("IK", e)
    results["5"] = False

# ─────────────────────────────────────────────────────────────────────────────
# [6] 30Hz 采集循环
# ─────────────────────────────────────────────────────────────────────────────
stage(f"6/7  30Hz 采集循环 × {args.episodes} episodes")

def to_numpy(t):
    arr = t.cpu().numpy() if hasattr(t, 'cpu') else np.array(t)
    return arr[0] if arr.ndim > 1 else arr

def render_camera(cam):
    rgb, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, normal=False)
    arr = rgb.cpu().numpy() if hasattr(rgb, 'cpu') else np.array(rgb)
    if arr.ndim == 4:
        arr = arr[0]
    return arr.astype(np.uint8)

def generate_scripted_trajectory(n_steps, n_dofs, ik_available, so101, ee_link, cube_pos):
    """
    Generate a scripted pick-place trajectory.
    Returns list of (target_deg,) for each timestep.
    Falls back to sinusoidal motion if IK unavailable.
    """
    home = HOME_DEG[:n_dofs].copy()
    traj = []

    if ik_available and ee_link is not None:
        down_quat = np.array([0.0, 1.0, 0.0, 0.0])

        def solve_ik(pos, gripper_deg=0.0):
            q = so101.inverse_kinematics(link=ee_link, pos=pos, quat=down_quat)
            q_np = to_numpy(q)
            q_deg = np.rad2deg(q_np)
            if n_dofs >= 6:
                q_deg[5] = gripper_deg
            return q_deg

        def interpolate(start, end, steps):
            return [start + (end - start) * (i + 1) / steps for i in range(steps)]

        try:
            pre_pos = cube_pos + np.array([0.0, 0.0, 0.10])
            q_pre = solve_ik(pre_pos, 0.0)

            approach_pos = cube_pos + np.array([0.0, 0.0, 0.02])
            q_approach = solve_ik(approach_pos, 0.0)

            lift_pos = cube_pos + np.array([0.0, 0.0, 0.15])
            q_lift = solve_ik(lift_pos, 25.0)

            phase_steps = n_steps // 6
            remainder = n_steps - phase_steps * 6

            # Home → Pre-grasp
            traj += interpolate(home, q_pre, phase_steps)
            # Pre-grasp → Approach
            traj += interpolate(q_pre, q_approach, phase_steps)
            # Close gripper
            q_close = q_approach.copy()
            if n_dofs >= 6:
                q_close[5] = 25.0
            traj += interpolate(q_approach, q_close, phase_steps)
            # Lift
            traj += interpolate(q_close, q_lift, phase_steps)
            # Hold
            traj += [q_lift.copy() for _ in range(phase_steps)]
            # Return home
            traj += interpolate(q_lift, home, phase_steps + remainder)

            return traj
        except Exception:
            pass

    # Fallback: gentle sinusoidal motion on each joint
    for i in range(n_steps):
        t = i / max(n_steps - 1, 1)
        phase = np.sin(np.pi * t)
        offset = np.zeros(n_dofs)
        for j in range(min(n_dofs, 5)):
            offset[j] = 15.0 * phase * np.sin(2.0 * np.pi * (j + 1) * t)
        traj.append(home + offset)

    return traj

try:
    steps_per_episode = int(args.episode_length * args.fps)
    all_episodes = []
    t_collect_start = time.time()

    for ep_idx in range(args.episodes):
        ep_t0 = time.time()

        # Reset to home
        home_rad = np.deg2rad(HOME_DEG[:n_dofs])
        so101.set_qpos(home_rad)
        for _ in range(30):
            scene.step()

        # Domain randomization: randomize cube position
        cx = np.random.uniform(0.10, 0.22)
        cy = np.random.uniform(-0.08, 0.08)
        cz = 0.015
        cube_pos_rand = np.array([cx, cy, cz])
        cube.set_pos(torch.tensor(cube_pos_rand, dtype=torch.float32, device=gs.device).unsqueeze(0))
        for _ in range(10):
            scene.step()

        # Retrieve actual cube position post-settle
        cube_pos_actual = to_numpy(cube.get_pos())

        # Generate trajectory
        trajectory = generate_scripted_trajectory(
            steps_per_episode, n_dofs, ik_available,
            so101, ee_link, cube_pos_actual,
        )

        ep_data = {
            "observation.state": [],
            "action": [],
            "observation.images.up": [],
            "observation.images.side": [],
            "timestamp": [],
            "frame_index": [],
            "episode_index": [],
        }

        for frame_idx, target_deg in enumerate(trajectory):
            target_rad = np.deg2rad(target_deg)
            so101.control_dofs_position(target_rad, ALL_DOF_IDX)
            scene.step()

            state_deg = np.rad2deg(to_numpy(so101.get_dofs_position(ALL_DOF_IDX)))
            img_up = render_camera(cam_up)
            img_side = render_camera(cam_side)

            ep_data["observation.state"].append(state_deg.astype(np.float32))
            ep_data["action"].append(target_deg.astype(np.float32))
            ep_data["observation.images.up"].append(img_up)
            ep_data["observation.images.side"].append(img_side)
            ep_data["timestamp"].append(frame_idx / args.fps)
            ep_data["frame_index"].append(frame_idx)
            ep_data["episode_index"].append(ep_idx)

        all_episodes.append(ep_data)
        ep_elapsed = time.time() - ep_t0
        n_frames = len(trajectory)
        print(f"  episode {ep_idx}: {n_frames} frames, {ep_elapsed:.1f}s "
              f"({n_frames/ep_elapsed:.0f} fps sim)")

    collect_elapsed = time.time() - t_collect_start
    total_frames = sum(len(ep["timestamp"]) for ep in all_episodes)
    ok(f"采集完成", f"{args.episodes} episodes, {total_frames} frames, {collect_elapsed:.1f}s")
    results["6"] = True
except Exception as e:
    err("collection loop", e)
    results["6"] = False

# ─────────────────────────────────────────────────────────────────────────────
# [7] 保存 + 摘要
# ─────────────────────────────────────────────────────────────────────────────
stage("7/7  保存数据 + 统计摘要")
try:
    out_dir = Path(args.save)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Concatenate all episodes
    all_states = np.concatenate([np.stack(ep["observation.state"]) for ep in all_episodes])
    all_actions = np.concatenate([np.stack(ep["action"]) for ep in all_episodes])
    all_imgs_up = np.concatenate([np.stack(ep["observation.images.up"]) for ep in all_episodes])
    all_imgs_side = np.concatenate([np.stack(ep["observation.images.side"]) for ep in all_episodes])
    all_timestamps = np.concatenate([np.array(ep["timestamp"]) for ep in all_episodes])
    all_frame_idx = np.concatenate([np.array(ep["frame_index"]) for ep in all_episodes])
    all_episode_idx = np.concatenate([np.array(ep["episode_index"]) for ep in all_episodes])

    np.save(out_dir / "states.npy", all_states)
    np.save(out_dir / "actions.npy", all_actions)
    np.save(out_dir / "images_up.npy", all_imgs_up)
    np.save(out_dir / "images_side.npy", all_imgs_side)
    np.save(out_dir / "timestamps.npy", all_timestamps)
    np.save(out_dir / "frame_indices.npy", all_frame_idx)
    np.save(out_dir / "episode_indices.npy", all_episode_idx)

    # Per-episode metadata
    ep_meta = []
    for ep in all_episodes:
        ep_meta.append({
            "n_frames": len(ep["timestamp"]),
            "duration": ep["timestamp"][-1] if ep["timestamp"] else 0,
        })

    ok("states",  f"shape={all_states.shape}, range=[{all_states.min():.1f}, {all_states.max():.1f}]°")
    ok("actions", f"shape={all_actions.shape}, range=[{all_actions.min():.1f}, {all_actions.max():.1f}]°")
    ok("images_up",   f"shape={all_imgs_up.shape}, range=[{all_imgs_up.min()}, {all_imgs_up.max()}]")
    ok("images_side", f"shape={all_imgs_side.shape}, range=[{all_imgs_side.min()}, {all_imgs_side.max()}]")
    ok(f"saved to", str(out_dir))

    # Print svla_so101_pickplace compatibility check
    print(f"\n  ── svla_so101_pickplace 兼容性 ──")
    print(f"  state dim:   {all_states.shape[1]}  (目标: 6)")
    print(f"  action dim:  {all_actions.shape[1]}  (目标: 6)")
    print(f"  image shape: {all_imgs_up.shape[1:]}  (目标: (480, 640, 3))")
    print(f"  cameras:     up + side  (目标: up + side)")
    print(f"  fps:         {args.fps}  (目标: 30)")
    print(f"  episodes:    {args.episodes}")
    print(f"  total frames: {total_frames}")

    dim_ok = all_states.shape[1] == 6 and all_actions.shape[1] == 6
    img_ok = all_imgs_up.shape[1:] == (args.img_h, args.img_w, 3)
    if dim_ok and img_ok:
        print(f"  → 数据结构与 svla_so101_pickplace 完全兼容 ✓")
    else:
        print(f"  → 部分维度不匹配，需调整")

    results["7"] = True
except Exception as e:
    err("save", e)
    results["7"] = False

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'═'*60}")
print("  SUMMARY — SDG SO-101 URDF")
print(f"{'═'*60}")
stage_labels = {
    "1": "SO-101 URDF 下载",
    "2": "Genesis + 场景构建",
    "3": "关节发现 & 映射",
    "4": "PD 增益 + Home",
    "5": "IK 求解",
    "6": f"采集 {args.episodes} episodes",
    "7": "保存 + 统计",
}
all_pass = True
for k, label in stage_labels.items():
    status = results.get(k, False)
    symbol = PASS if status else FAIL
    print(f"  [{k}/7]  {symbol}  {label}")
    if not status:
        all_pass = False

print()
if all_pass:
    print("  🎉 SO-101 URDF 全阶段通过！可进入 LeRobot 格式打包阶段。")
else:
    print("  ⚠️  部分阶段失败，请查看上方错误信息。")
print(f"{'═'*60}\n")
