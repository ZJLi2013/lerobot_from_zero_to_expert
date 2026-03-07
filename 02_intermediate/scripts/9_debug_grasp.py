"""
Minimal debug grasp runner for SO-101.

Runs a single fixed-offset episode in a clean scene (no auto-tune trials),
exports dense PNGs around approach+close for visual inspection.

Usage:
  python 9_debug_grasp.py --xml /path/to/v3.xml --exp-id E49_clean \
      --offset-x 0.008 --offset-y -0.004 --offset-z -0.01 \
      --save /output
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def ensure_display():
    if os.environ.get("DISPLAY"):
        return
    xvfb = subprocess.run(["which", "Xvfb"], capture_output=True)
    if xvfb.returncode != 0:
        return
    print("[display] Starting Xvfb :99 ...")
    proc = subprocess.Popen(
        ["Xvfb", ":99", "-screen", "0", "1280x1024x24", "-ac", "+extension", "GLX"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.environ["DISPLAY"] = ":99"
    time.sleep(2)
    if proc.poll() is None:
        print(f"[display] Xvfb started (PID={proc.pid})")


def to_numpy(t):
    arr = t.cpu().numpy() if hasattr(t, "cpu") else np.array(t)
    return arr[0] if arr.ndim > 1 else arr


def render_camera(cam):
    rgb, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, normal=False)
    arr = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.array(rgb)
    if arr.ndim == 4:
        arr = arr[0]
    return arr.astype(np.uint8)


def save_png(arr, path):
    try:
        from PIL import Image
        Image.fromarray(arr).save(path)
    except ImportError:
        import imageio.v2 as imageio
        imageio.imwrite(path, arr)


def find_xml(user_path):
    if user_path:
        p = Path(user_path)
        if p.exists():
            return p
    for c in [
        Path("assets/so101_new_calib.xml"),
        Path("02_intermediate/scripts/assets/so101_new_calib.xml"),
    ]:
        if c.exists():
            return c
    try:
        from huggingface_hub import snapshot_download
        d = snapshot_download(
            repo_type="dataset", repo_id="Genesis-Intelligence/assets",
            allow_patterns="SO101/*", max_workers=1,
        )
        p = Path(d) / "SO101" / "so101_new_calib.xml"
        if p.exists():
            return p
    except Exception:
        pass
    return None


JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]
HOME_DEG = np.array([0.0, -30.0, 90.0, -60.0, 0.0, 0.0], dtype=np.float32)
IK_QUAT_DOWN = np.array([0.0, 1.0, 0.0, 0.0])


def lerp(a, b, n):
    a, b = np.array(a), np.array(b)
    return [a + (b - a) * i / max(n - 1, 1) for i in range(n)]


def main():
    ensure_display()

    p = argparse.ArgumentParser()
    p.add_argument("--xml", default=None)
    p.add_argument("--exp-id", default="E49_debug")
    p.add_argument("--save", default="/output")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--episode-length", type=float, default=8.0)
    p.add_argument("--gripper-open", type=float, default=15.0)
    p.add_argument("--gripper-close", type=float, default=-10.0)
    p.add_argument("--close-hold-steps", type=int, default=50)
    p.add_argument("--approach-z", type=float, default=0.012)
    p.add_argument("--offset-x", type=float, default=0.008)
    p.add_argument("--offset-y", type=float, default=-0.004)
    p.add_argument("--offset-z", type=float, default=-0.01)
    p.add_argument("--cube-x", type=float, default=0.16)
    p.add_argument("--cube-y", type=float, default=0.0)
    p.add_argument("--cube-z", type=float, default=0.015)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--run-auto-tune-before", action="store_true",
                   help="Run 25 auto-tune trials before the episode (to reproduce E48-like scene state)")
    p.add_argument("--auto-tune-x", default="-0.008,-0.004,0.0,0.004,0.008")
    p.add_argument("--auto-tune-y", default="-0.008,-0.004,0.0,0.004,0.008")
    p.add_argument("--auto-tune-z", default="-0.010")
    args = p.parse_args()

    xml_path = find_xml(args.xml)
    if not xml_path:
        print("XML not found"); sys.exit(1)
    print(f"XML: {xml_path}")

    import torch
    import genesis as gs
    import genesis.utils.geom as gu

    backend = gs.cpu if args.cpu else gs.gpu
    gs.init(backend=backend, logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / args.fps, substeps=4),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True, enable_joint_limit=True,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    cube = scene.add_entity(
        gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=(args.cube_x, args.cube_y, args.cube_z)),
        surface=gs.surfaces.Default(color=(1.0, 0.3, 0.3, 1.0)),
    )
    so101 = scene.add_entity(gs.morphs.MJCF(file=str(xml_path), pos=(0, 0, 0)))

    cam_up = scene.add_camera(
        res=(640, 480), pos=(0.42, 0.34, 0.26), lookat=(0.15, 0.0, 0.08), fov=38, GUI=False,
    )
    cam_side = scene.add_camera(
        res=(640, 480), pos=(0.5, -0.4, 0.3), lookat=(0.15, 0.0, 0.1), fov=45, GUI=False,
    )
    scene.build()
    print("Scene built")

    home_rad = np.deg2rad(HOME_DEG)
    dof_idx = [so101.get_joint(n).dof_idx_local for n in JOINT_NAMES]
    n_dofs = len(dof_idx)

    so101.set_dofs_kp(np.array([500, 500, 400, 400, 300, 200], dtype=np.float32), dof_idx)
    so101.set_dofs_kv(np.array([50, 50, 40, 40, 30, 20], dtype=np.float32), dof_idx)
    so101.set_qpos(home_rad)
    so101.control_dofs_position(home_rad, dof_idx)
    for _ in range(60):
        scene.step()

    ee_link = so101.get_link("grasp_center")

    def solve_ik(pos, gripper_deg, seed_rad=None):
        if seed_rad is None:
            seed_rad = home_rad
        so101.set_qpos(seed_rad)
        result = so101.inverse_kinematics(
            link=ee_link, pos=np.array(pos, dtype=np.float32),
            quat=np.array(IK_QUAT_DOWN, dtype=np.float32),
        )
        q = np.rad2deg(to_numpy(result))[:n_dofs].tolist()
        q[5] = gripper_deg
        return q

    def reset_scene(cube_pos, settle=30):
        so101.set_qpos(home_rad)
        so101.control_dofs_position(home_rad, dof_idx)
        so101.zero_all_dofs_velocity()
        cube.set_pos(torch.tensor(cube_pos, dtype=torch.float32, device=gs.device).unsqueeze(0))
        cube.set_quat(torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=gs.device).unsqueeze(0))
        cube.zero_all_dofs_velocity()
        for _ in range(settle):
            scene.step()

    cube_pos = np.array([args.cube_x, args.cube_y, args.cube_z])

    # optionally run auto-tune trials to reproduce E48-like scene state
    if args.run_auto_tune_before:
        x_cands = [float(x) for x in args.auto_tune_x.split(",")]
        y_cands = [float(x) for x in args.auto_tune_y.split(",")]
        z_cands = [float(x) for x in args.auto_tune_z.split(",")]
        total = len(x_cands) * len(y_cands) * len(z_cands)
        print(f"Running {total} auto-tune trials before episode...")
        for oz in z_cands:
            for ox in x_cands:
                for oy in y_cands:
                    reset_scene(cube_pos, settle=20)
                    off = np.array([ox, oy, oz])
                    pos_app = cube_pos + off + np.array([0, 0, args.approach_z])
                    pos_pre = cube_pos + off + np.array([0, 0, 0.10])
                    q_pre = solve_ik(pos_pre, args.gripper_open, seed_rad=home_rad)
                    prev_rad = np.deg2rad(np.array(q_pre, dtype=np.float32))
                    for i in range(6):
                        frac = (i + 1) / 6
                        z = 0.10 + (args.approach_z - 0.10) * frac
                        pos = cube_pos + off + np.array([0, 0, z])
                        q = solve_ik(pos, args.gripper_open, seed_rad=prev_rad)
                        prev_rad = np.deg2rad(np.array(q, dtype=np.float32))
                    q_app = q
                    q_close = q_app.copy()
                    q_close[5] = args.gripper_close
                    for t_deg in lerp(home_rad * 0 + np.array(q_pre), q_app, 15) + lerp(q_app, q_close, 8):
                        t_rad = np.deg2rad(np.array(t_deg, dtype=np.float32))
                        so101.control_dofs_position(t_rad, dof_idx)
                        scene.step()
        print(f"Auto-tune trials done.")

    # --- clean episode ---
    print(f"\nRunning episode with offset=[{args.offset_x}, {args.offset_y}, {args.offset_z}]")
    reset_scene(cube_pos, settle=30)

    off = np.array([args.offset_x, args.offset_y, args.offset_z])
    pos_pre = cube_pos + off + np.array([0, 0, 0.10])
    pos_approach = cube_pos + off + np.array([0, 0, args.approach_z])

    q_pre = solve_ik(pos_pre, args.gripper_open, seed_rad=home_rad)
    prev_rad = np.deg2rad(np.array(q_pre, dtype=np.float32))
    descent_wps = []
    for i in range(6):
        frac = (i + 1) / 6
        z = 0.10 + (args.approach_z - 0.10) * frac
        pos = cube_pos + off + np.array([0, 0, z])
        wp = solve_ik(pos, args.gripper_open, seed_rad=prev_rad)
        descent_wps.append(wp)
        prev_rad = np.deg2rad(np.array(wp, dtype=np.float32))

    q_approach = descent_wps[-1]
    q_close = q_approach.copy()
    q_close[5] = args.gripper_close

    q_close_rad = np.deg2rad(np.array(q_close, dtype=np.float32))
    lift_wps = []
    prev_rad = q_close_rad
    for i in range(4):
        frac = (i + 1) / 4
        z = args.approach_z + (0.15 - args.approach_z) * frac
        pos = cube_pos + off + np.array([0, 0, z])
        wp = solve_ik(pos, args.gripper_close, seed_rad=prev_rad)
        lift_wps.append(wp)
        prev_rad = np.deg2rad(np.array(wp, dtype=np.float32))

    steps = int(args.episode_length * args.fps)
    traj, phases = [], []
    n_move = max(15, steps // 8)
    traj += lerp(HOME_DEG.tolist(), q_pre, n_move)
    phases += ["move_pre"] * n_move
    steps_per_wp = max(3, steps // 48)
    prev = q_pre
    for wp in descent_wps:
        traj += lerp(prev, wp, steps_per_wp)
        phases += ["approach"] * steps_per_wp
        prev = wp
    n_close = max(8, steps // 12)
    traj += lerp(q_approach, q_close, n_close)
    phases += ["close"] * n_close
    n_hold = args.close_hold_steps
    traj += [q_close[:] for _ in range(n_hold)]
    phases += ["close_hold"] * n_hold
    steps_per_lift = max(5, steps // 32)
    prev = q_close
    for wp in lift_wps:
        traj += lerp(prev, wp, steps_per_lift)
        phases += ["lift"] * steps_per_lift
        prev = wp

    reset_scene(cube_pos, settle=30)

    images = []
    frame_phases = []
    for fi, (target_deg, phase) in enumerate(zip(traj, phases)):
        target_rad = np.deg2rad(np.array(target_deg, dtype=np.float32))
        so101.control_dofs_position(target_rad, dof_idx)
        scene.step()
        up_img = render_camera(cam_up)
        side_img = render_camera(cam_side)
        images.append(np.concatenate([up_img, side_img], axis=1))
        frame_phases.append(phase)

    cube_z_final = float(to_numpy(cube.get_pos())[2])
    print(f"cube_z final = {cube_z_final:.4f}")

    # export PNGs: approach last 4 frames + all close + close_hold first 10 + lift first 4
    out_dir = Path(args.save) / args.exp_id / "debug_pngs"
    out_dir.mkdir(parents=True, exist_ok=True)

    approach_idx = [i for i, p in enumerate(frame_phases) if p == "approach"]
    close_idx = [i for i, p in enumerate(frame_phases) if p == "close"]
    hold_idx = [i for i, p in enumerate(frame_phases) if p == "close_hold"]
    lift_idx = [i for i, p in enumerate(frame_phases) if p == "lift"]

    export_frames = set()
    if approach_idx:
        export_frames.update(approach_idx[-4:])
    export_frames.update(close_idx)
    export_frames.update(hold_idx[:10])
    if lift_idx:
        export_frames.update(lift_idx[:4])

    for i in sorted(export_frames):
        save_png(images[i], out_dir / f"f{i:03d}_{frame_phases[i]}.png")

    print(f"Exported {len(export_frames)} PNGs to {out_dir}")
    print(f"  approach frames: {approach_idx[-4:] if approach_idx else []}")
    print(f"  close frames: {close_idx}")
    print(f"  auto-tune-before: {args.run_auto_tune_before}")


if __name__ == "__main__":
    main()
