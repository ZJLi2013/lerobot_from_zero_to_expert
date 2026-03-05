"""
SO-101 grasp experiment runner (1 episode focused).

Purpose:
- Validate "gripper points down but does not grasp" fixes.
- Run one controlled episode and export:
  - npy data
  - rrd visualization
  - metrics.json (grasp success indicators)
"""
import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np


def ensure_display():
    if os.environ.get("DISPLAY"):
        return
    xvfb = subprocess.run(["which", "Xvfb"], capture_output=True)
    if xvfb.returncode != 0:
        return
    subprocess.Popen(
        ["Xvfb", ":99", "-screen", "0", "1280x1024x24", "-ac", "+extension", "GLX"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.environ["DISPLAY"] = ":99"
    time.sleep(1.5)


def to_numpy(t):
    arr = t.cpu().numpy() if hasattr(t, "cpu") else np.array(t)
    return arr[0] if arr.ndim > 1 else arr


def render_camera(cam):
    rgb, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, normal=False)
    arr = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.array(rgb)
    if arr.ndim == 4:
        arr = arr[0]
    return arr.astype(np.uint8)


def joint_comfort_score(q_rad, limits):
    score = 0.0
    for i in range(min(len(q_rad), len(limits))):
        lo, hi = limits[i]
        span = hi - lo
        if span < 1e-6:
            continue
        mid = (lo + hi) / 2.0
        score += ((q_rad[i] - mid) / (span / 2.0)) ** 2
    return score


def parse_csv_floats(text):
    vals = []
    for item in text.split(","):
        item = item.strip()
        if item:
            vals.append(float(item))
    return vals


def download_urdf(target_dir: Path):
    hf_base = "https://huggingface.co/haixuantao/dora-bambot/resolve/main/URDF"
    urdf_filename = "so101.urdf"
    stl_files = [
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
    target_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = target_dir / "assets"
    assets_dir.mkdir(exist_ok=True)
    urdf_path = target_dir / urdf_filename
    if not urdf_path.exists():
        urllib.request.urlretrieve(f"{hf_base}/{urdf_filename}", str(urdf_path))
    for stl in stl_files:
        p = assets_dir / stl
        if not p.exists():
            urllib.request.urlretrieve(f"{hf_base}/assets/{stl}", str(p))
    return urdf_path


def main():
    ensure_display()
    parser = argparse.ArgumentParser(description="SO-101 1-episode grasp experiment")
    parser.add_argument("--exp-id", default="E1_baseline_tcp")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episode-length", type=float, default=8.0)
    parser.add_argument("--img-w", type=int, default=640)
    parser.add_argument("--img-h", type=int, default=480)
    parser.add_argument("--save", default="/output")
    parser.add_argument("--urdf-dir", default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--euler", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--ee-link", default="gripperframe")
    parser.add_argument("--cube-x-min", type=float, default=0.12)
    parser.add_argument("--cube-x-max", type=float, default=0.20)
    parser.add_argument("--cube-y-min", type=float, default=-0.05)
    parser.add_argument("--cube-y-max", type=float, default=0.05)
    parser.add_argument("--gripper-open", type=float, default=70.0)
    parser.add_argument("--gripper-close", type=float, default=20.0)
    parser.add_argument("--close-hold-steps", type=int, default=12)
    parser.add_argument("--lift-threshold", type=float, default=0.01)
    parser.add_argument("--grasp-offset-x", type=float, default=0.0)
    parser.add_argument("--grasp-offset-y", type=float, default=0.0)
    parser.add_argument("--grasp-offset-z", type=float, default=0.0)
    parser.add_argument("--auto-tune-offset", action="store_true")
    parser.add_argument("--offset-x-candidates", default="-0.008,-0.004,0.0,0.004,0.008")
    parser.add_argument("--offset-y-candidates", default="-0.010,-0.005,0.0,0.005,0.010")
    args = parser.parse_args()

    import torch
    import genesis as gs
    import rerun as rr

    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    limits = np.array(
        [
            [-1.91986, 1.91986],
            [-1.74533, 1.74533],
            [-1.74533, 1.5708],
            [-1.65806, 1.65806],
            [-2.74385, 2.84121],
            [-0.174533, 1.74533],
        ]
    )
    base_offset = (0.163, 0.168, 0.0)
    kp = np.array([500.0, 500.0, 400.0, 400.0, 300.0, 200.0])
    kv = np.array([50.0, 50.0, 40.0, 40.0, 30.0, 20.0])

    urdf_dir = Path(args.urdf_dir) if args.urdf_dir else Path(__file__).resolve().parent / "models" / "so101_urdf"
    urdf_path = urdf_dir / "so101.urdf" if args.no_download else download_urdf(urdf_dir)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF missing: {urdf_path}")

    backend = gs.cpu if args.cpu else gs.gpu
    gs.init(backend=backend, logging_level="warning")
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
        gs.morphs.URDF(file=str(urdf_path), pos=base_offset, euler=tuple(args.euler), fixed=True)
    )
    cam_up = scene.add_camera(res=(args.img_w, args.img_h), pos=(0.0, 0.0, 0.55), lookat=(0.0, 0.0, 0.08), fov=60, GUI=False)
    cam_side = scene.add_camera(res=(args.img_w, args.img_h), pos=(0.35, -0.30, 0.20), lookat=(0.0, 0.0, 0.10), fov=55, GUI=False)
    scene.build()

    n_dofs = so101.n_dofs
    dof_idx = np.arange(n_dofs)
    so101.set_dofs_kp(kp[:n_dofs], dof_idx)
    so101.set_dofs_kv(kv[:n_dofs], dof_idx)

    # Prefer tcp frame, fallback to gripper
    ee_link = None
    for candidate in [args.ee_link, "gripperframe", "gripper", "moving_jaw_so101_v1"]:
        try:
            ee_link = so101.get_link(candidate)
            ee_name = candidate
            break
        except Exception:
            continue
    if ee_link is None:
        raise RuntimeError("No available ee link")

    # Probe quaternion search in front workspace
    ik_candidates = [
        ("identity", np.array([1, 0, 0, 0], dtype=np.float64)),
        ("180x", np.array([0, 1, 0, 0], dtype=np.float64)),
        ("180y", np.array([0, 0, 1, 0], dtype=np.float64)),
        ("180z", np.array([0, 0, 0, 1], dtype=np.float64)),
        ("90x", np.array([0.7071, 0.7071, 0, 0], dtype=np.float64)),
        ("-90x", np.array([0.7071, -0.7071, 0, 0], dtype=np.float64)),
        ("90y", np.array([0.7071, 0, 0.7071, 0], dtype=np.float64)),
        ("-90y", np.array([0.7071, 0, -0.7071, 0], dtype=np.float64)),
    ]
    probe_pos = np.array([0.14, 0.0, 0.08])
    best_name, best_quat, best_score = None, None, float("inf")
    for name, quat in ik_candidates:
        try:
            q = to_numpy(so101.inverse_kinematics(link=ee_link, pos=probe_pos, quat=quat))
            within = True
            for i in range(min(n_dofs, len(limits))):
                if q[i] < limits[i, 0] - 0.01 or q[i] > limits[i, 1] + 0.01:
                    within = False
                    break
            if not within:
                continue
            s = joint_comfort_score(q, limits[:n_dofs])
            if s < best_score:
                best_name, best_quat, best_score = name, quat.copy(), s
        except Exception:
            continue
    if best_quat is None:
        best_name, best_quat = "fallback_180x", np.array([0, 1, 0, 0], dtype=np.float64)

    # Home from reachable front-up point
    try:
        home_rad = to_numpy(so101.inverse_kinematics(link=ee_link, pos=np.array([0.14, 0.0, 0.16]), quat=best_quat))
    except Exception:
        home_rad = np.zeros(n_dofs)
    home_deg = np.rad2deg(home_rad)

    all_episodes = []
    metrics = {
        "exp_id": args.exp_id,
        "ee_link": ee_name,
        "ik_quat_name": best_name,
        "ik_quat": best_quat.tolist(),
        "gripper_open_deg": args.gripper_open,
        "gripper_close_deg": args.gripper_close,
        "close_hold_steps": args.close_hold_steps,
        "lift_threshold_m": args.lift_threshold,
        "grasp_offset_initial": [args.grasp_offset_x, args.grasp_offset_y, args.grasp_offset_z],
        "episodes": [],
    }

    steps_per_episode = int(args.episode_length * args.fps)
    for ep in range(args.episodes):
        # reset
        so101.set_qpos(home_rad)
        so101.control_dofs_position(home_rad, dof_idx)
        for _ in range(50):
            scene.step()

        # randomized cube in front workspace
        cube_pos = np.array(
            [
                np.random.uniform(args.cube_x_min, args.cube_x_max),
                np.random.uniform(args.cube_y_min, args.cube_y_max),
                0.015,
            ]
        )
        cube.set_pos(torch.tensor(cube_pos, dtype=torch.float32, device=gs.device).unsqueeze(0))
        for _ in range(15):
            scene.step()
        cube_pos = to_numpy(cube.get_pos())

        def solve_ik(pos, grip_deg):
            q = to_numpy(so101.inverse_kinematics(link=ee_link, pos=pos, quat=best_quat))
            q_deg = np.rad2deg(q)
            if n_dofs >= 6:
                q_deg[5] = grip_deg
            return q_deg

        def build_key_poses(offset_x, offset_y, offset_z):
            off = np.array([offset_x, offset_y, offset_z], dtype=np.float64)
            q_home_local = home_deg.copy()
            q_pre_local = solve_ik(cube_pos + off + np.array([0.0, 0.0, 0.10]), args.gripper_open)
            q_approach_local = solve_ik(cube_pos + off + np.array([0.0, 0.0, 0.025]), args.gripper_open)
            q_close_local = q_approach_local.copy()
            if n_dofs >= 6:
                q_close_local[5] = args.gripper_close
            q_lift_local = solve_ik(cube_pos + off + np.array([0.0, 0.0, 0.16]), args.gripper_close)
            return q_home_local, q_pre_local, q_approach_local, q_close_local, q_lift_local

        def run_trial(offset_x, offset_y, offset_z):
            # Reset robot and cube so each candidate is comparable.
            so101.set_qpos(home_rad)
            so101.control_dofs_position(home_rad, dof_idx)
            cube.set_pos(torch.tensor(cube_pos, dtype=torch.float32, device=gs.device).unsqueeze(0))
            for _ in range(20):
                scene.step()

            q_home_t, q_pre_t, q_approach_t, q_close_t, q_lift_t = build_key_poses(offset_x, offset_y, offset_z)

            def lerp(a, b, n):
                return [a + (b - a) * (i + 1) / max(n, 1) for i in range(n)]

            trial_traj = []
            trial_phases = []
            trial_traj += lerp(q_home_t, q_pre_t, 15); trial_phases += ["move_pre"] * 15
            trial_traj += lerp(q_pre_t, q_approach_t, 15); trial_phases += ["approach"] * 15
            trial_traj += lerp(q_approach_t, q_close_t, 10); trial_phases += ["close"] * 10
            trial_traj += [q_close_t.copy() for _ in range(args.close_hold_steps)]; trial_phases += ["close_hold"] * args.close_hold_steps
            trial_traj += lerp(q_close_t, q_lift_t, 20); trial_phases += ["lift"] * 20

            z_before = None
            z_after = None
            for target_deg, phase in zip(trial_traj, trial_phases):
                target_rad = np.deg2rad(np.array(target_deg, dtype=np.float32))
                so101.control_dofs_position(target_rad, dof_idx)
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

        # Optional automatic offset search (x/y grid).
        chosen_offset = np.array([args.grasp_offset_x, args.grasp_offset_y, args.grasp_offset_z], dtype=np.float64)
        if args.auto_tune_offset:
            x_candidates = parse_csv_floats(args.offset_x_candidates)
            y_candidates = parse_csv_floats(args.offset_y_candidates)
            best_delta = -1e9
            best_xy = (chosen_offset[0], chosen_offset[1])
            search_log = []
            for ox in x_candidates:
                for oy in y_candidates:
                    try:
                        delta = run_trial(ox, oy, chosen_offset[2])
                    except Exception:
                        delta = -1.0
                    search_log.append({"offset_x": ox, "offset_y": oy, "lift_delta": float(delta)})
                    if delta > best_delta:
                        best_delta = delta
                        best_xy = (ox, oy)
            chosen_offset[0] = best_xy[0]
            chosen_offset[1] = best_xy[1]
            metrics["auto_tune"] = {
                "enabled": True,
                "best_offset_xy": [float(chosen_offset[0]), float(chosen_offset[1])],
                "best_lift_delta": float(best_delta),
                "search_log": search_log,
            }
        else:
            metrics["auto_tune"] = {"enabled": False}

        metrics["selected_grasp_offset"] = [float(chosen_offset[0]), float(chosen_offset[1]), float(chosen_offset[2])]
        q_home, q_pre, q_approach, q_close, q_lift = build_key_poses(
            chosen_offset[0], chosen_offset[1], chosen_offset[2]
        )

        # Build trajectory with explicit hold after close
        def lerp(a, b, n):
            return [a + (b - a) * (i + 1) / max(n, 1) for i in range(n)]

        n1 = max(20, steps_per_episode // 8)
        n2 = max(20, steps_per_episode // 8)
        n3 = max(10, steps_per_episode // 12)
        n4 = args.close_hold_steps
        n5 = max(25, steps_per_episode // 8)
        consumed = n1 + n2 + n3 + n4 + n5
        n6 = max(0, steps_per_episode - consumed)

        trajectory = []
        labels = []
        trajectory += lerp(q_home, q_pre, n1); labels += ["move_pre"] * n1
        trajectory += lerp(q_pre, q_approach, n2); labels += ["approach"] * n2
        trajectory += lerp(q_approach, q_close, n3); labels += ["close"] * n3
        trajectory += [q_close.copy() for _ in range(n4)]; labels += ["close_hold"] * n4
        trajectory += lerp(q_close, q_lift, n5); labels += ["lift"] * n5
        trajectory += lerp(q_lift, q_home, n6); labels += ["return"] * n6

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
            target_rad = np.deg2rad(np.array(target_deg, dtype=np.float32))
            so101.control_dofs_position(target_rad, dof_idx)
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
        metrics["episodes"].append(
            {
                "episode": ep,
                "cube_z_before_close": cube_z_before_close,
                "cube_z_after_lift": cube_z_after_lift,
                "cube_lift_delta": lift_delta,
                "grasp_success": grasp_success,
            }
        )
        print(
            f"[ep {ep}] success={grasp_success} delta_z={lift_delta:.4f}m "
            f"(before={cube_z_before_close:.4f}, after={cube_z_after_lift:.4f})"
        )

    # Save npy
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
    np.save(out_dir / "states.npy", all_states)
    np.save(out_dir / "actions.npy", all_actions)
    np.save(out_dir / "images_up.npy", all_imgs_up)
    np.save(out_dir / "images_side.npy", all_imgs_side)
    np.save(out_dir / "timestamps.npy", all_ts)
    np.save(out_dir / "frame_indices.npy", all_fi)
    np.save(out_dir / "episode_indices.npy", all_ei)
    np.save(out_dir / "cube_z.npy", all_cube_z)

    # Save rrd
    rrd_path = out_dir / f"improved_sdg_{args.exp_id}.rrd"
    rr.init(f"so101_sdg_{args.exp_id}", spawn=False)
    n_joints = min(all_states.shape[1], len(joint_names))
    for i in range(len(all_states)):
        rr.set_time("frame_index", sequence=int(all_fi[i]))
        rr.set_time("timestamp", timestamp=float(all_ts[i]))
        rr.log("observation.images.up", rr.Image(all_imgs_up[i]))
        rr.log("observation.images.side", rr.Image(all_imgs_side[i]))
        rr.log("object/cube_z", rr.Scalars(float(all_cube_z[i])))
        for j in range(n_joints):
            jn = joint_names[j]
            rr.log(f"state/{jn}", rr.Scalars(float(all_states[i, j])))
            rr.log(f"action/{jn}", rr.Scalars(float(all_actions[i, j])))
    rr.save(str(rrd_path))

    # Save metrics
    metrics["state_range_deg"] = [float(all_states.min()), float(all_states.max())]
    metrics["action_range_deg"] = [float(all_actions.min()), float(all_actions.max())]
    metrics["output_dir"] = str(out_dir)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_dir}")
    print(f"RRD: {rrd_path}")


if __name__ == "__main__":
    main()
