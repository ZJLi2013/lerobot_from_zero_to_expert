"""
SO-101 grasp experiment runner — MJCF version.

Purpose:
- Debug and tune grasp parameters using the official SO-101 MJCF model.
- Auto-offset search to find best gripper→cube alignment.
- Export npy + rrd + metrics.json for each experiment.

Usage:
  python 3_grasp_experiment.py --exp-id E1 --episodes 1 --episode-length 6
  python 3_grasp_experiment.py --exp-id E2_auto --auto-tune-offset
  python 3_grasp_experiment.py --xml /path/to/so101.xml --exp-id E3
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


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


def to_numpy(t):
    arr = t.cpu().numpy() if hasattr(t, "cpu") else np.array(t)
    return arr[0] if arr.ndim > 1 else arr


def render_camera(cam):
    rgb, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, normal=False)
    arr = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.array(rgb)
    if arr.ndim == 4:
        arr = arr[0]
    return arr.astype(np.uint8)


def parse_csv_floats(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


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


# ── Constants (proven working with MJCF from 2_collect.py) ────────────────────

JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]
HOME_DEG = np.array([0.0, -30.0, 90.0, -60.0, 0.0, 0.0], dtype=np.float32)
IK_QUAT_DOWN = np.array([0.0, 1.0, 0.0, 0.0])
KP = np.array([500.0, 500.0, 400.0, 400.0, 300.0, 200.0])
KV = np.array([50.0, 50.0, 40.0, 40.0, 30.0, 20.0])


def stage(name):
    print(f"\n{'─'*60}\n  [{name}]\n{'─'*60}")


def main():
    ensure_display()

    parser = argparse.ArgumentParser(description="SO-101 grasp experiment (MJCF)")
    parser.add_argument("--exp-id", default="E1_mjcf")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episode-length", type=float, default=8.0)
    parser.add_argument("--img-w", type=int, default=640)
    parser.add_argument("--img-h", type=int, default=480)
    parser.add_argument("--save", default="/output")
    parser.add_argument("--xml", default=None, help="Path to so101_new_calib.xml")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gripper-open", type=float, default=0.0,
                        help="Gripper open angle in degrees")
    parser.add_argument("--gripper-close", type=float, default=25.0,
                        help="Gripper close angle in degrees")
    parser.add_argument("--close-hold-steps", type=int, default=12)
    parser.add_argument("--lift-threshold", type=float, default=0.01)
    parser.add_argument("--approach-z", type=float, default=0.02,
                        help="Z offset above cube center for approach (m)")
    parser.add_argument("--grasp-offset-x", type=float, default=0.0)
    parser.add_argument("--grasp-offset-y", type=float, default=0.0)
    parser.add_argument("--grasp-offset-z", type=float, default=0.0)
    parser.add_argument("--auto-tune-offset", action="store_true")
    parser.add_argument("--offset-x-candidates", default="-0.01,-0.005,0.0,0.005,0.01")
    parser.add_argument("--offset-y-candidates", default="-0.01,-0.005,0.0,0.005,0.01")
    parser.add_argument("--offset-z-candidates", default="-0.02,-0.01,0.0,0.01")
    args = parser.parse_args()

    # ── [1] Locate MJCF ──────────────────────────────────────────────────────
    stage("1/6  定位 SO-101 MJCF")
    xml_path = find_so101_xml(args.xml)
    if xml_path is None:
        print("  ✗ so101_new_calib.xml not found")
        print("    pip install huggingface_hub   OR   --xml /path/to/so101.xml")
        sys.exit(1)
    print(f"  ✓ {xml_path}")

    # ── [2] Genesis + Scene ───────────────────────────────────────────────────
    stage("2/6  Genesis + Scene")
    import torch
    import genesis as gs

    backend = gs.cpu if args.cpu else gs.gpu
    gs.init(backend=backend, logging_level="warning")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / args.fps, substeps=4),
        rigid_options=gs.options.RigidOptions(enable_collision=True, enable_joint_limit=True),
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
        pos=(0.0, 0.0, 0.7), lookat=(0.15, 0.0, 0.0), fov=55, GUI=False,
    )
    cam_side = scene.add_camera(
        res=(args.img_w, args.img_h),
        pos=(0.5, -0.4, 0.3), lookat=(0.15, 0.0, 0.1), fov=45, GUI=False,
    )
    scene.build()
    print("  ✓ scene built")

    # ── [3] Joints + EE + PD + Home ──────────────────────────────────────────
    stage("3/6  Joints + PD + Home")
    n_dofs = so101.n_dofs
    dof_idx = np.arange(n_dofs)
    print(f"  n_dofs = {n_dofs}")

    for j in so101.joints:
        print(f"    joint: {j.name}")

    ee_link = None
    ee_name = None
    for candidate in ["gripper_link", "gripperframe", "gripper", "Fixed_Jaw"]:
        try:
            ee_link = so101.get_link(candidate)
            ee_name = candidate
            print(f"  ✓ EE link = {candidate}")
            break
        except Exception:
            pass
    if ee_link is None:
        print("  ✗ EE link not found")
        sys.exit(1)

    kp = KP[:n_dofs]
    kv = KV[:n_dofs]
    so101.set_dofs_kp(kp, dof_idx)
    so101.set_dofs_kv(kv, dof_idx)

    home_deg = HOME_DEG[:n_dofs]
    home_rad = np.deg2rad(home_deg)
    so101.set_qpos(home_rad)
    so101.control_dofs_position(home_rad, dof_idx)
    for _ in range(60):
        scene.step()

    cur_deg = np.rad2deg(to_numpy(so101.get_dofs_position(dof_idx)))
    home_err = np.abs(cur_deg - home_deg).mean()
    print(f"  HOME = {home_deg.tolist()}")
    print(f"  tracking: mean_err={home_err:.2f}°")

    # IK sanity check
    try:
        q_test = so101.inverse_kinematics(
            link=ee_link, pos=np.array([0.15, 0.0, 0.08]), quat=IK_QUAT_DOWN
        )
        print(f"  ✓ IK sanity check passed")
    except Exception as e:
        print(f"  ✗ IK failed: {e}")
        sys.exit(1)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def solve_ik(pos, grip_deg):
        q = to_numpy(so101.inverse_kinematics(link=ee_link, pos=pos, quat=IK_QUAT_DOWN))
        q_deg = np.rad2deg(q)
        if n_dofs >= 6:
            q_deg[5] = grip_deg
        return q_deg

    def lerp(a, b, n):
        return [a + (b - a) * (i + 1) / max(n, 1) for i in range(n)]

    def cartesian_descent(cube_pos, off, z_from, z_to, grip_deg, n_waypoints=5):
        """Solve IK at multiple intermediate heights for straight-line descent."""
        waypoints = []
        for i in range(n_waypoints):
            frac = (i + 1) / n_waypoints
            z = z_from + (z_to - z_from) * frac
            pos = cube_pos + off + np.array([0.0, 0.0, z])
            waypoints.append(solve_ik(pos, grip_deg))
        return waypoints

    def reset_scene(cube_pos, settle_steps=30):
        """Full reset: position + velocity for robot and cube."""
        so101.set_qpos(home_rad)
        so101.control_dofs_position(home_rad, dof_idx)
        so101.set_dofs_velocity(np.zeros(n_dofs, dtype=np.float32), dof_idx)
        cube.set_pos(torch.tensor(cube_pos, dtype=torch.float32, device=gs.device).unsqueeze(0))
        cube.set_quat(torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=gs.device).unsqueeze(0))
        for _ in range(settle_steps):
            scene.step()

    def build_trajectory(cube_pos, offset_x, offset_y, offset_z, total_steps):
        """Build trajectory with Cartesian-space descent to avoid lateral swing."""
        off = np.array([offset_x, offset_y, offset_z])
        q_pre = solve_ik(cube_pos + off + np.array([0.0, 0.0, 0.10]), args.gripper_open)
        q_approach = solve_ik(
            cube_pos + off + np.array([0.0, 0.0, args.approach_z]), args.gripper_open
        )
        q_close = q_approach.copy()
        if n_dofs >= 6:
            q_close[5] = args.gripper_close
        q_lift = solve_ik(cube_pos + off + np.array([0.0, 0.0, 0.15]), args.gripper_close)

        descent_wps = cartesian_descent(
            cube_pos, off, 0.10, args.approach_z, args.gripper_open, n_waypoints=6
        )

        traj = []
        phases = []

        n_move = max(15, total_steps // 8)
        traj += lerp(home_deg.copy(), q_pre, n_move)
        phases += ["move_pre"] * n_move

        steps_per_wp = max(3, total_steps // (8 * len(descent_wps)))
        prev = q_pre
        for wp in descent_wps:
            traj += lerp(prev, wp, steps_per_wp)
            phases += ["approach"] * steps_per_wp
            prev = wp

        n_close = max(8, total_steps // 12)
        traj += lerp(q_approach, q_close, n_close)
        phases += ["close"] * n_close

        n_hold = args.close_hold_steps
        traj += [q_close.copy() for _ in range(n_hold)]
        phases += ["close_hold"] * n_hold

        lift_wps = cartesian_descent(
            cube_pos, off, args.approach_z, 0.15, args.gripper_close, n_waypoints=4
        )
        steps_per_lift = max(5, total_steps // (8 * len(lift_wps)))
        prev = q_close
        for wp in lift_wps:
            traj += lerp(prev, wp, steps_per_lift)
            phases += ["lift"] * steps_per_lift
            prev = wp

        consumed = len(traj)
        n_return = max(0, total_steps - consumed)
        if n_return > 0:
            traj += lerp(q_lift, home_deg.copy(), n_return)
            phases += ["return"] * n_return

        return traj, phases

    def run_trial(cube_pos, offset_x, offset_y, offset_z):
        """Quick trial: reset → Cartesian descent → close → lift → measure delta_z."""
        reset_scene(cube_pos, settle_steps=20)

        traj, phases = build_trajectory(cube_pos, offset_x, offset_y, offset_z, total_steps=90)

        z_before = None
        z_after = None
        for target_deg, phase in zip(traj, phases):
            target_rad_t = np.deg2rad(np.array(target_deg, dtype=np.float32))
            so101.control_dofs_position(target_rad_t, dof_idx)
            scene.step()
            z_now = float(to_numpy(cube.get_pos())[2])
            if phase == "close" and z_before is None:
                z_before = z_now
            if phase == "lift":
                z_after = z_now

        if z_before is None:
            z_before = float(to_numpy(cube.get_pos())[2])
        if z_after is None:
            z_after = float(to_numpy(cube.get_pos())[2])
        return z_after - z_before

    # ── [4] Auto-tune offset (optional) ──────────────────────────────────────
    stage("4/6  Offset 调参")
    steps_per_episode = int(args.episode_length * args.fps)

    metrics = {
        "exp_id": args.exp_id,
        "model": "MJCF",
        "xml_path": str(xml_path),
        "ee_link": ee_name,
        "ik_quat": IK_QUAT_DOWN.tolist(),
        "home_deg": home_deg.tolist(),
        "gripper_open_deg": args.gripper_open,
        "gripper_close_deg": args.gripper_close,
        "approach_z": args.approach_z,
        "close_hold_steps": args.close_hold_steps,
        "lift_threshold_m": args.lift_threshold,
        "episodes": [],
    }

    all_episodes = []

    for ep in range(args.episodes):
        # Reset
        so101.set_qpos(home_rad)
        so101.control_dofs_position(home_rad, dof_idx)
        for _ in range(50):
            scene.step()

        # Randomize cube
        cube_pos = np.array([
            np.random.uniform(0.12, 0.20),
            np.random.uniform(-0.05, 0.05),
            0.015,
        ])
        cube.set_pos(torch.tensor(cube_pos, dtype=torch.float32, device=gs.device).unsqueeze(0))
        for _ in range(15):
            scene.step()
        cube_pos = to_numpy(cube.get_pos())
        print(f"  cube pos = [{cube_pos[0]:.4f}, {cube_pos[1]:.4f}, {cube_pos[2]:.4f}]")

        # Optional auto-tune offset (grid search over x/y/z)
        chosen_offset = np.array(
            [args.grasp_offset_x, args.grasp_offset_y, args.grasp_offset_z], dtype=np.float64
        )
        if args.auto_tune_offset:
            x_cands = parse_csv_floats(args.offset_x_candidates)
            y_cands = parse_csv_floats(args.offset_y_candidates)
            z_cands = parse_csv_floats(args.offset_z_candidates)
            best_delta = -1e9
            best_xyz = (chosen_offset[0], chosen_offset[1], chosen_offset[2])
            search_log = []
            total = len(x_cands) * len(y_cands) * len(z_cands)
            tried = 0
            for oz in z_cands:
                for ox in x_cands:
                    for oy in y_cands:
                        tried += 1
                        try:
                            delta = run_trial(cube_pos, ox, oy, oz)
                        except Exception:
                            delta = -1.0
                        search_log.append({
                            "ox": ox, "oy": oy, "oz": oz, "delta_z": float(delta)
                        })
                        if delta > best_delta:
                            best_delta = delta
                            best_xyz = (ox, oy, oz)
                        tag = "★" if delta > args.lift_threshold else " "
                        if tried <= 20 or delta > args.lift_threshold:
                            print(
                                f"    {tag} [{tried}/{total}] "
                                f"ox={ox:+.3f} oy={oy:+.3f} oz={oz:+.3f} → Δz={delta:+.4f}m"
                            )
            chosen_offset[:] = best_xyz
            print(f"  ✓ best offset=({best_xyz[0]:+.3f}, {best_xyz[1]:+.3f}, {best_xyz[2]:+.3f}) Δz={best_delta:+.4f}m")
            metrics["auto_tune"] = {
                "enabled": True,
                "best_offset": [float(v) for v in best_xyz],
                "best_lift_delta": float(best_delta),
                "search_log": search_log,
            }
        else:
            print(f"  offset = ({chosen_offset[0]:+.3f}, {chosen_offset[1]:+.3f}, {chosen_offset[2]:+.3f})")
            metrics["auto_tune"] = {"enabled": False}

        metrics["selected_grasp_offset"] = [float(v) for v in chosen_offset]

        # ── [5] Full episode collection with chosen offset ────────────────────
        stage(f"5/6  数据采集 ep {ep}")

        reset_scene(cube_pos, settle_steps=30)

        trajectory, labels = build_trajectory(
            cube_pos, chosen_offset[0], chosen_offset[1], chosen_offset[2],
            total_steps=steps_per_episode,
        )

        ep_data = {
            "observation.state": [],
            "action": [],
            "observation.images.up": [],
            "observation.images.side": [],
            "timestamp": [],
            "frame_index": [],
            "episode_index": [],
            "cube_z": [],
            "phase": [],
        }
        cube_z_before_close = None
        cube_z_after_lift = None

        for fi, (target_deg, phase) in enumerate(zip(trajectory, labels)):
            target_rad_f = np.deg2rad(np.array(target_deg, dtype=np.float32))
            so101.control_dofs_position(target_rad_f, dof_idx)
            scene.step()

            state_deg = np.rad2deg(to_numpy(so101.get_dofs_position(dof_idx)))
            cube_z = float(to_numpy(cube.get_pos())[2])
            if phase == "close" and cube_z_before_close is None:
                cube_z_before_close = cube_z
            if phase == "lift":
                cube_z_after_lift = cube_z

            ep_data["observation.state"].append(state_deg.astype(np.float32))
            ep_data["action"].append(np.array(target_deg, dtype=np.float32))
            ep_data["observation.images.up"].append(render_camera(cam_up))
            ep_data["observation.images.side"].append(render_camera(cam_side))
            ep_data["timestamp"].append(fi / args.fps)
            ep_data["frame_index"].append(fi)
            ep_data["episode_index"].append(ep)
            ep_data["cube_z"].append(cube_z)
            ep_data["phase"].append(phase)

        all_episodes.append(ep_data)
        if cube_z_before_close is None:
            cube_z_before_close = float(to_numpy(cube.get_pos())[2])
        if cube_z_after_lift is None:
            cube_z_after_lift = float(to_numpy(cube.get_pos())[2])
        lift_delta = cube_z_after_lift - cube_z_before_close
        grasp_success = 1 if lift_delta > args.lift_threshold else 0
        metrics["episodes"].append({
            "episode": ep,
            "cube_pos": cube_pos.tolist(),
            "cube_z_before_close": cube_z_before_close,
            "cube_z_after_lift": cube_z_after_lift,
            "cube_lift_delta": lift_delta,
            "grasp_success": grasp_success,
        })
        print(
            f"  [ep {ep}] grasp={'✓' if grasp_success else '✗'} "
            f"delta_z={lift_delta:.4f}m "
            f"(before={cube_z_before_close:.4f}, after={cube_z_after_lift:.4f})"
        )

    # ── [6] Save ──────────────────────────────────────────────────────────────
    stage("6/6  保存")
    import rerun as rr

    out_dir = Path(args.save) / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    all_states = np.concatenate([np.stack(e["observation.state"]) for e in all_episodes])
    all_actions = np.concatenate([np.stack(e["action"]) for e in all_episodes])
    all_imgs_up = np.concatenate([np.stack(e["observation.images.up"]) for e in all_episodes])
    all_imgs_side = np.concatenate([np.stack(e["observation.images.side"]) for e in all_episodes])
    all_ts = np.concatenate([np.array(e["timestamp"]) for e in all_episodes])
    all_fi = np.concatenate([np.array(e["frame_index"]) for e in all_episodes])
    all_ei = np.concatenate([np.array(e["episode_index"]) for e in all_episodes])
    all_cube_z = np.concatenate([np.array(e["cube_z"], dtype=np.float32) for e in all_episodes])

    for name, arr in [
        ("states", all_states), ("actions", all_actions),
        ("images_up", all_imgs_up), ("images_side", all_imgs_side),
        ("timestamps", all_ts), ("frame_indices", all_fi),
        ("episode_indices", all_ei), ("cube_z", all_cube_z),
    ]:
        np.save(out_dir / f"{name}.npy", arr)
        print(f"  {name}: {arr.shape}")

    rrd_path = out_dir / f"grasp_{args.exp_id}.rrd"
    rr.init(f"so101_grasp_{args.exp_id}", spawn=False)
    n_joints = min(all_states.shape[1], len(JOINT_NAMES))
    for i in range(len(all_states)):
        rr.set_time("frame_index", sequence=int(all_fi[i]))
        rr.set_time("timestamp", timestamp=float(all_ts[i]))
        rr.log("observation.images.up", rr.Image(all_imgs_up[i]))
        rr.log("observation.images.side", rr.Image(all_imgs_side[i]))
        rr.log("object/cube_z", rr.Scalars(float(all_cube_z[i])))
        for j in range(n_joints):
            jn = JOINT_NAMES[j]
            rr.log(f"state/{jn}", rr.Scalars(float(all_states[i, j])))
            rr.log(f"action/{jn}", rr.Scalars(float(all_actions[i, j])))
    rr.save(str(rrd_path))
    rrd_mb = rrd_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ {rrd_path} ({rrd_mb:.1f} MB)")

    metrics["state_range_deg"] = [float(all_states.min()), float(all_states.max())]
    metrics["action_range_deg"] = [float(all_actions.min()), float(all_actions.max())]
    metrics["output_dir"] = str(out_dir)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  ✓ metrics.json")

    # Summary
    total_success = sum(1 for e in metrics["episodes"] if e["grasp_success"])
    total_eps = len(metrics["episodes"])
    print(f"\n{'═'*60}")
    print(f"  SUMMARY — {args.exp_id}")
    print(f"{'═'*60}")
    print(f"  Grasp success: {total_success}/{total_eps}")
    if metrics.get("auto_tune", {}).get("enabled"):
        best = metrics["auto_tune"]
        print(f"  Best offset: {best['best_offset']}, Δz={best['best_lift_delta']:.4f}m")
    print(f"  Output: {out_dir}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
