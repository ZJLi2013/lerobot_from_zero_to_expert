"""
Gated pre-grasp experiment aligned with 3/12 trajectory style.

Baseline:
- Uses the same approach trajectory construction style as 12_approach_only_diagnostic.py.

Gated:
- Keeps the same move_pre and descend skeleton,
- Adds a pre_grasp_level loop at safe height,
- Allows descend only after |z_moving - z_fixed| < level_tol_z (or timeout).
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
        return
    xvfb = subprocess.run(["which", "Xvfb"], capture_output=True)
    if xvfb.returncode != 0:
        return
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


def save_rgb_png(arr, path):
    try:
        from PIL import Image

        Image.fromarray(arr).save(path)
    except ImportError:
        import imageio.v2 as imageio

        imageio.imwrite(path, arr)


def parse_csv_floats(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def find_so101_xml(user_path=None):
    if user_path:
        p = Path(user_path)
        if p.exists():
            return p
        return None
    candidates = [
        Path("assets/so101_new_calib_v4.xml"),
        Path("02_intermediate/scripts/assets/so101_new_calib_v4.xml"),
        Path("assets/so101_new_calib.xml"),
        Path("02_intermediate/scripts/assets/so101_new_calib.xml"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def box_tilt_deg(cube_quat):
    cube_rot = quat_to_rotmat(cube_quat)
    cube_z_axis = cube_rot @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cosv = abs(float(np.clip(np.dot(cube_z_axis / (np.linalg.norm(cube_z_axis) + 1e-9), np.array([0.0, 0.0, 1.0])), -1.0, 1.0)))
    return float(np.degrees(np.arccos(cosv)))


def stage(name):
    print(f"\n{'-' * 60}\n[{name}]\n{'-' * 60}")


HOME_DEG = np.array([0.0, -30.0, 90.0, -60.0, 0.0, 0.0], dtype=np.float32)
KP = np.array([500.0, 500.0, 400.0, 400.0, 300.0, 200.0], dtype=np.float32)
KV = np.array([50.0, 50.0, 40.0, 40.0, 30.0, 20.0], dtype=np.float32)


def main():
    ensure_display()
    parser = argparse.ArgumentParser(description="Gated pre_grasp experiment aligned to 3/12")
    parser.add_argument("--exp-id", default="gated_exp3_aligned")
    parser.add_argument("--xml", default=None)
    parser.add_argument("--save", default="/output")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sim-substeps", type=int, default=4)
    parser.add_argument("--sim-dt", type=float, default=None)
    parser.add_argument("--trial-steps", type=int, default=90)
    parser.add_argument("--approach-hold-steps", type=int, default=1)
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument("--cube-x", type=float, default=0.16)
    parser.add_argument("--cube-y", type=float, default=0.0)
    parser.add_argument("--cube-z", type=float, default=0.015)
    parser.add_argument("--cube-friction", type=float, default=1.5)
    parser.add_argument("--gripper-open", type=float, default=20.0)
    parser.add_argument("--grasp-offset-x", type=float, default=0.0)
    parser.add_argument("--grasp-offset-y", type=float, default=0.0)
    parser.add_argument("--grasp-offset-z", type=float, default=-0.01)
    parser.add_argument("--approach-z", type=float, default=0.012)
    parser.add_argument("--z-safe-offset", type=float, default=0.10)
    parser.add_argument("--level-max-steps", type=int, default=20)
    parser.add_argument("--level-tol-z", type=float, default=0.002)
    parser.add_argument("--level-step-y", type=float, default=0.001)
    parser.add_argument("--export-last-frames", type=int, default=10)
    args = parser.parse_args()

    stage("1/5 setup")
    xml_path = find_so101_xml(args.xml)
    if xml_path is None:
        print("xml not found")
        sys.exit(1)
    print(f"xml: {xml_path}")

    import torch
    import genesis as gs

    backend = gs.cpu if args.cpu else gs.gpu
    gs.init(backend=backend, logging_level="warning")
    sim_dt = args.sim_dt if args.sim_dt is not None else 1.0 / args.fps

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=sim_dt, substeps=args.sim_substeps),
        rigid_options=gs.options.RigidOptions(enable_collision=True, enable_joint_limit=True, box_box_detection=True),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    cube = scene.add_entity(
        morph=gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=(args.cube_x, args.cube_y, args.cube_z)),
        material=gs.materials.Rigid(friction=args.cube_friction),
        surface=gs.surfaces.Default(color=(1.0, 0.3, 0.3, 1.0)),
    )
    so101 = scene.add_entity(gs.morphs.MJCF(file=str(xml_path), pos=(0.0, 0.0, 0.0)))
    cam_up = scene.add_camera(res=(640, 480), pos=(0.42, 0.34, 0.26), lookat=(0.15, 0.0, 0.08), fov=38, GUI=False)
    cam_side = scene.add_camera(res=(640, 480), pos=(0.5, -0.4, 0.3), lookat=(0.15, 0.0, 0.1), fov=45, GUI=False)
    scene.build()

    dof_idx = np.arange(so101.n_dofs)
    so101.set_dofs_kp(KP[: so101.n_dofs], dof_idx)
    so101.set_dofs_kv(KV[: so101.n_dofs], dof_idx)
    home_deg = HOME_DEG[: so101.n_dofs]
    home_rad = np.deg2rad(home_deg)

    ee_link = so101.get_link("grasp_center")
    fixed_link = so101.get_link("gripper")
    moving_link = so101.get_link("moving_jaw_so101_v1")
    cube_init = np.array([args.cube_x, args.cube_y, args.cube_z], dtype=np.float64)
    off = np.array([args.grasp_offset_x, args.grasp_offset_y, args.grasp_offset_z], dtype=np.float64)

    def reset_scene():
        so101.set_qpos(home_rad)
        so101.control_dofs_position(home_rad, dof_idx)
        so101.zero_all_dofs_velocity()
        cube.set_pos(torch.tensor(cube_init, dtype=torch.float32, device=gs.device).unsqueeze(0))
        cube.set_quat(torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=gs.device).unsqueeze(0))
        cube.zero_all_dofs_velocity()
        for _ in range(args.settle_steps):
            scene.step()

    def solve_ik_seeded(pos, grip_deg, seed_rad=None):
        if seed_rad is None:
            seed_rad = home_rad
        q = to_numpy(
            so101.inverse_kinematics(
                link=ee_link,
                pos=np.array(pos, dtype=np.float32),
                quat=None,
                init_qpos=seed_rad,
                max_solver_iters=50,
                damping=0.02,
            )
        )
        q_deg = np.rad2deg(q)
        if so101.n_dofs >= 6:
            q_deg[5] = grip_deg
        return q_deg

    def lerp(a, b, n):
        a = np.array(a, dtype=np.float64)
        b = np.array(b, dtype=np.float64)
        return [a + (b - a) * (i + 1) / max(n, 1) for i in range(n)]

    def build_baseline_trajectory():
        # exactly mirrors 12_approach_only_diagnostic build_approach_trajectory structure
        pos_pre = cube_init + off + np.array([0.0, 0.0, 0.10], dtype=np.float64)
        q_pre = solve_ik_seeded(pos_pre, args.gripper_open, seed_rad=home_rad)
        q_pre_rad = np.deg2rad(np.array(q_pre, dtype=np.float32))
        n_descent_wps = 6
        descent_wps = []
        prev_rad = q_pre_rad
        for i in range(n_descent_wps):
            frac = (i + 1) / n_descent_wps
            z = 0.10 + (args.approach_z - 0.10) * frac
            pos = cube_init + off + np.array([0.0, 0.0, z], dtype=np.float64)
            wp = solve_ik_seeded(pos, args.gripper_open, seed_rad=prev_rad)
            descent_wps.append(wp)
            prev_rad = np.deg2rad(np.array(wp, dtype=np.float32))

        q_approach = descent_wps[-1]
        traj, phases = [], []
        n_move = max(15, args.trial_steps // 8)
        traj += lerp(home_deg.copy(), q_pre, n_move)
        phases += ["move_pre"] * n_move

        steps_per_wp = max(3, args.trial_steps // (8 * n_descent_wps))
        prev = q_pre
        for wp in descent_wps:
            traj += lerp(prev, wp, steps_per_wp)
            phases += ["approach"] * steps_per_wp
            prev = wp
        if args.approach_hold_steps > 0:
            traj += [q_approach.copy() for _ in range(args.approach_hold_steps)]
            phases += ["approach_hold"] * args.approach_hold_steps
        return traj, phases

    def run_baseline(out_dir):
        traj, phases = build_baseline_trajectory()
        frame_buffer = []
        keep_idx = [i for i, ph in enumerate(phases) if ph == "approach"]
        keep_idx = set(keep_idx[-args.export_last_frames :])
        for fi, (target_deg, phase) in enumerate(zip(traj, phases)):
            target_rad = np.deg2rad(np.array(target_deg, dtype=np.float32))
            so101.control_dofs_position(target_rad, dof_idx)
            scene.step()
            if fi in keep_idx:
                stitched = np.concatenate([render_camera(cam_up), render_camera(cam_side)], axis=1)
                frame_buffer.append((fi, phase, stitched))
        cube_pos_final = to_numpy(cube.get_pos())
        shift = cube_pos_final - cube_init
        tilt = box_tilt_deg(to_numpy(cube.get_quat()))
        out_dir.mkdir(parents=True, exist_ok=True)
        for fi, phase, img in frame_buffer:
            save_rgb_png(img, out_dir / f"f{fi:03d}_{phase}.png")
        return {
            "approach_contact": float(np.linalg.norm(shift)),
            "approach_shift_xyz": [float(v) for v in shift.tolist()],
            "box_tilt_deg": float(tilt),
            "frame_dir": str(out_dir),
            "last_indices": sorted(list(keep_idx)),
        }

    def run_gated(out_dir):
        # aligned skeleton: same move_pre + same approach descend count
        pos_pre = cube_init + off + np.array([0.0, 0.0, 0.10], dtype=np.float64)
        q_pre = solve_ik_seeded(pos_pre, args.gripper_open, seed_rad=home_rad)
        z_end = cube_init[2] + off[2] + args.approach_z
        z_safe = cube_init[2] + off[2] + args.z_safe_offset
        n_descent_wps = 6
        n_move = max(15, args.trial_steps // 8)
        steps_per_wp = max(3, args.trial_steps // (8 * n_descent_wps))

        traj = []
        phases = []
        traj += lerp(home_deg.copy(), q_pre, n_move)
        phases += ["move_pre"] * n_move

        # execute move_pre first
        for target_deg in traj:
            so101.control_dofs_position(np.deg2rad(np.array(target_deg, dtype=np.float32)), dof_idx)
            scene.step()

        # level loop at z_safe
        prev_rad = np.deg2rad(np.array(q_pre, dtype=np.float32))
        x_t = float(cube_init[0] + off[0])
        y_t = float(cube_init[1] + off[1])
        level_history = []
        leveled = False
        for k in range(args.level_max_steps):
            q = solve_ik_seeded([x_t, y_t, z_safe], args.gripper_open, seed_rad=prev_rad)
            prev_rad = np.deg2rad(np.array(q, dtype=np.float32))
            so101.control_dofs_position(np.deg2rad(np.array(q, dtype=np.float32)), dof_idx)
            scene.step()
            traj.append(q)
            phases.append("pre_grasp_level")
            z_fixed = float(to_numpy(fixed_link.get_pos())[2])
            z_moving = float(to_numpy(moving_link.get_pos())[2])
            dz = z_moving - z_fixed
            level_history.append({"step": int(k), "target_xy": [x_t, y_t], "z_fixed": z_fixed, "z_moving": z_moving, "dz": float(dz)})
            if abs(dz) < args.level_tol_z:
                leveled = True
                break
            y_t += -np.sign(dz) * args.level_step_y

        # descend phase with same count/style as baseline
        descent_wps = []
        for i in range(n_descent_wps):
            frac = (i + 1) / n_descent_wps
            z = z_safe + (z_end - z_safe) * frac
            q = solve_ik_seeded([x_t, y_t, z], args.gripper_open, seed_rad=prev_rad)
            descent_wps.append(q)
            prev_rad = np.deg2rad(np.array(q, dtype=np.float32))

        prev = traj[-1] if traj else q_pre
        for wp in descent_wps:
            seg = lerp(prev, wp, steps_per_wp)
            for target_deg in seg:
                so101.control_dofs_position(np.deg2rad(np.array(target_deg, dtype=np.float32)), dof_idx)
                scene.step()
                traj.append(target_deg)
                phases.append("approach")
            prev = wp

        if args.approach_hold_steps > 0:
            for _ in range(args.approach_hold_steps):
                so101.control_dofs_position(np.deg2rad(np.array(prev, dtype=np.float32)), dof_idx)
                scene.step()
                traj.append(prev.copy())
                phases.append("approach_hold")

        frame_buffer = []
        keep_idx = [i for i, ph in enumerate(phases) if ph == "approach"]
        keep_idx = set(keep_idx[-args.export_last_frames :])
        for fi, (_, phase) in enumerate(zip(traj, phases)):
            if fi in keep_idx:
                stitched = np.concatenate([render_camera(cam_up), render_camera(cam_side)], axis=1)
                frame_buffer.append((fi, phase, stitched))

        cube_pos_final = to_numpy(cube.get_pos())
        shift = cube_pos_final - cube_init
        tilt = box_tilt_deg(to_numpy(cube.get_quat()))
        out_dir.mkdir(parents=True, exist_ok=True)
        for fi, phase, img in frame_buffer:
            save_rgb_png(img, out_dir / f"f{fi:03d}_{phase}.png")
        return {
            "approach_contact": float(np.linalg.norm(shift)),
            "approach_shift_xyz": [float(v) for v in shift.tolist()],
            "box_tilt_deg": float(tilt),
            "frame_dir": str(out_dir),
            "last_indices": sorted(list(keep_idx)),
            "leveled": bool(leveled),
            "final_level_dz": float(level_history[-1]["dz"]) if level_history else None,
            "level_history": level_history,
        }

    stage("2/5 baseline")
    out_root = Path(args.save) / args.exp_id
    reset_scene()
    baseline = run_baseline(out_root / "baseline_pngs")

    stage("3/5 gated")
    reset_scene()
    gated = run_gated(out_root / "gated_pngs")

    stage("4/5 summary")
    summary = {
        "exp_id": args.exp_id,
        "xml_path": str(xml_path),
        "cube_pose": [float(v) for v in cube_init.tolist()],
        "offset_xyz": [float(v) for v in off.tolist()],
        "baseline": baseline,
        "gated": gated,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "baseline_contact": baseline["approach_contact"],
                "gated_contact": gated["approach_contact"],
                "baseline_tilt": baseline["box_tilt_deg"],
                "gated_tilt": gated["box_tilt_deg"],
                "leveled": gated["leveled"],
                "final_level_dz": gated["final_level_dz"],
                "output": str(out_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""
Gated pre-grasp experiment (baseline vs gated) for SO-101.

Purpose:
- Keep only the essential logic needed to validate the gated pre_grasp idea.
- Compare baseline direct descend vs gated pre_grasp_level + short_descend.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np


def ensure_display():
    if os.environ.get("DISPLAY"):
        return
    xvfb = subprocess.run(["which", "Xvfb"], capture_output=True)
    if xvfb.returncode != 0:
        return
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


def save_rgb_png(arr, path):
    try:
        from PIL import Image

        Image.fromarray(arr).save(path)
    except ImportError:
        import imageio.v2 as imageio

        imageio.imwrite(path, arr)


def find_so101_xml(user_path=None):
    if user_path:
        p = Path(user_path)
        if p.exists():
            return p
        return None
    candidates = [
        Path("assets/so101_new_calib_v4.xml"),
        Path("02_intermediate/scripts/assets/so101_new_calib_v4.xml"),
        Path("assets/so101_new_calib.xml"),
        Path("02_intermediate/scripts/assets/so101_new_calib.xml"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def stage(name):
    print(f"\n{'-' * 60}\n[{name}]\n{'-' * 60}")


def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def box_tilt_deg(cube_quat):
    r = quat_to_rotmat(cube_quat)
    z_axis = r @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cosv = abs(float(np.clip(np.dot(z_axis / (np.linalg.norm(z_axis) + 1e-9), np.array([0.0, 0.0, 1.0])), -1.0, 1.0)))
    return float(np.degrees(np.arccos(cosv)))


HOME_DEG = np.array([0.0, -30.0, 90.0, -60.0, 0.0, 0.0], dtype=np.float32)
KP = np.array([500.0, 500.0, 400.0, 400.0, 300.0, 200.0], dtype=np.float32)
KV = np.array([50.0, 50.0, 40.0, 40.0, 30.0, 20.0], dtype=np.float32)


def main():
    ensure_display()
    parser = argparse.ArgumentParser(description="SO-101 gated pre-grasp experiment")
    parser.add_argument("--exp-id", default="gated_pregrasp_v1")
    parser.add_argument("--xml", default=None)
    parser.add_argument("--save", default="/output")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sim-substeps", type=int, default=4)
    parser.add_argument("--sim-dt", type=float, default=None)
    parser.add_argument("--cube-x", type=float, default=0.16)
    parser.add_argument("--cube-y", type=float, default=0.0)
    parser.add_argument("--cube-z", type=float, default=0.015)
    parser.add_argument("--cube-friction", type=float, default=1.5)
    parser.add_argument("--gripper-open", type=float, default=20.0)
    parser.add_argument("--grasp-offset-x", type=float, default=0.0)
    parser.add_argument("--grasp-offset-y", type=float, default=0.0)
    parser.add_argument("--grasp-offset-z", type=float, default=-0.01)
    parser.add_argument("--z-safe", type=float, default=0.06)
    parser.add_argument("--approach-z", type=float, default=0.032)
    parser.add_argument("--level-max-steps", type=int, default=20)
    parser.add_argument("--level-tol-z", type=float, default=0.002)
    parser.add_argument("--level-step-y", type=float, default=0.001)
    parser.add_argument("--export-last-frames", type=int, default=10)
    parser.add_argument("--settle-steps", type=int, default=20)
    args = parser.parse_args()

    stage("1/5 setup")
    xml_path = find_so101_xml(args.xml)
    if xml_path is None:
        raise RuntimeError("SO-101 XML not found")
    print(f"xml: {xml_path}")

    import torch
    import genesis as gs

    backend = gs.cpu if args.cpu else gs.gpu
    gs.init(backend=backend, logging_level="warning")
    sim_dt = args.sim_dt if args.sim_dt is not None else 1.0 / args.fps
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=sim_dt, substeps=args.sim_substeps),
        rigid_options=gs.options.RigidOptions(enable_collision=True, enable_joint_limit=True, box_box_detection=True),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    cube = scene.add_entity(
        morph=gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=(args.cube_x, args.cube_y, args.cube_z)),
        material=gs.materials.Rigid(friction=args.cube_friction),
        surface=gs.surfaces.Default(color=(1.0, 0.3, 0.3, 1.0)),
    )
    so101 = scene.add_entity(gs.morphs.MJCF(file=str(xml_path), pos=(0.0, 0.0, 0.0)))
    cam_up = scene.add_camera(res=(640, 480), pos=(0.42, 0.34, 0.26), lookat=(0.15, 0.0, 0.08), fov=38, GUI=False)
    cam_side = scene.add_camera(res=(640, 480), pos=(0.5, -0.4, 0.3), lookat=(0.15, 0.0, 0.1), fov=45, GUI=False)
    scene.build()

    dof_idx = np.arange(so101.n_dofs)
    so101.set_dofs_kp(KP[: so101.n_dofs], dof_idx)
    so101.set_dofs_kv(KV[: so101.n_dofs], dof_idx)
    home_deg = HOME_DEG[: so101.n_dofs]
    home_rad = np.deg2rad(home_deg)

    ee_link = so101.get_link("grasp_center")
    fixed_link = so101.get_link("gripper")
    moving_link = so101.get_link("moving_jaw_so101_v1")
    cube_init = np.array([args.cube_x, args.cube_y, args.cube_z], dtype=np.float64)
    off = np.array([args.grasp_offset_x, args.grasp_offset_y, args.grasp_offset_z], dtype=np.float64)

    def reset_scene():
        so101.set_qpos(home_rad)
        so101.control_dofs_position(home_rad, dof_idx)
        so101.zero_all_dofs_velocity()
        cube.set_pos(torch.tensor(cube_init, dtype=torch.float32, device=gs.device).unsqueeze(0))
        cube.set_quat(torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=gs.device).unsqueeze(0))
        cube.zero_all_dofs_velocity()
        for _ in range(args.settle_steps):
            scene.step()

    def solve_ik(pos, grip_deg, seed=None):
        if seed is None:
            seed = home_rad
        q = to_numpy(
            so101.inverse_kinematics(
                link=ee_link,
                pos=np.array(pos, dtype=np.float32),
                quat=None,
                init_qpos=seed,
                max_solver_iters=50,
                damping=0.02,
            )
        )
        q_deg = np.rad2deg(q)
        if so101.n_dofs >= 6:
            q_deg[5] = grip_deg
        return q_deg

    def run_traj(traj, phases, out_dir):
        frame_buffer = []
        keep = [i for i, p in enumerate(phases) if p in ("baseline_descend", "short_descend")]
        keep = keep[-args.export_last_frames :]
        keep = set(keep)
        for i, (target_deg, phase) in enumerate(zip(traj, phases)):
            target_rad = np.deg2rad(np.array(target_deg, dtype=np.float32))
            so101.control_dofs_position(target_rad, dof_idx)
            scene.step()
            if i in keep:
                stitched = np.concatenate([render_camera(cam_up), render_camera(cam_side)], axis=1)
                frame_buffer.append((i, phase, stitched))
        cube_final = to_numpy(cube.get_pos())
        shift = cube_final - cube_init
        tilt = box_tilt_deg(to_numpy(cube.get_quat()))
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, phase, img in frame_buffer:
            save_rgb_png(img, out_dir / f"f{i:03d}_{phase}.png")
        return {
            "approach_contact": float(np.linalg.norm(shift)),
            "approach_shift_xyz": [float(v) for v in shift.tolist()],
            "box_tilt_deg": float(tilt),
            "frame_dir": str(out_dir),
        }

    stage("2/5 baseline trajectory")
    base_xy = cube_init[:2] + off[:2]
    z_pre = cube_init[2] + off[2] + 0.10
    z_end = cube_init[2] + off[2] + args.approach_z
    n_desc = 6
    traj_base = []
    phases_base = []
    q_pre = solve_ik([base_xy[0], base_xy[1], z_pre], args.gripper_open, seed=home_rad)
    traj_base.append(q_pre)
    phases_base.append("move_pre")
    prev = np.deg2rad(np.array(q_pre, dtype=np.float32))
    for i in range(n_desc):
        frac = (i + 1) / n_desc
        z = z_pre + (z_end - z_pre) * frac
        q = solve_ik([base_xy[0], base_xy[1], z], args.gripper_open, seed=prev)
        traj_base.append(q)
        phases_base.append("baseline_descend")
        prev = np.deg2rad(np.array(q, dtype=np.float32))

    stage("3/5 gated trajectory")
    traj_gate = []
    phases_gate = []
    q_pre_gate = solve_ik([base_xy[0], base_xy[1], args.z_safe], args.gripper_open, seed=home_rad)
    traj_gate.append(q_pre_gate)
    phases_gate.append("move_pre")
    prev_gate = np.deg2rad(np.array(q_pre_gate, dtype=np.float32))

    level_history = []
    x_t, y_t = float(base_xy[0]), float(base_xy[1])
    leveled = False
    for k in range(args.level_max_steps):
        q = solve_ik([x_t, y_t, args.z_safe], args.gripper_open, seed=prev_gate)
        traj_gate.append(q)
        phases_gate.append("pre_grasp_level")
        prev_gate = np.deg2rad(np.array(q, dtype=np.float32))
        so101.control_dofs_position(np.deg2rad(np.array(q, dtype=np.float32)), dof_idx)
        scene.step()

        z_fixed = float(to_numpy(fixed_link.get_pos())[2])
        z_moving = float(to_numpy(moving_link.get_pos())[2])
        dz = z_moving - z_fixed
        level_history.append({"step": int(k), "target_xy": [x_t, y_t], "z_fixed": z_fixed, "z_moving": z_moving, "dz": float(dz)})
        if abs(dz) < args.level_tol_z:
            leveled = True
            break
        y_t += -np.sign(dz) * args.level_step_y

    n_short = 4
    for i in range(n_short):
        frac = (i + 1) / n_short
        z = args.z_safe + (z_end - args.z_safe) * frac
        q = solve_ik([x_t, y_t, z], args.gripper_open, seed=prev_gate)
        traj_gate.append(q)
        phases_gate.append("short_descend")
        prev_gate = np.deg2rad(np.array(q, dtype=np.float32))

    stage("4/5 run and compare")
    out_root = Path(args.save) / args.exp_id
    reset_scene()
    baseline_metrics = run_traj(traj_base, phases_base, out_root / "baseline_pngs")
    reset_scene()
    gated_metrics = run_traj(traj_gate, phases_gate, out_root / "gated_pngs")

    stage("5/5 save")
    summary = {
        "exp_id": args.exp_id,
        "xml_path": str(xml_path),
        "cube_pose": [float(v) for v in cube_init.tolist()],
        "offset_xyz": [float(v) for v in off.tolist()],
        "baseline": {
            "approach_z": float(args.approach_z),
            **baseline_metrics,
        },
        "gated": {
            "z_safe": float(args.z_safe),
            "level_tol_z": float(args.level_tol_z),
            "level_step_y": float(args.level_step_y),
            "level_max_steps": int(args.level_max_steps),
            "leveled": bool(leveled),
            "final_level_dz": float(level_history[-1]["dz"]) if level_history else None,
            "level_history": level_history,
            **gated_metrics,
        },
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "baseline_contact": summary["baseline"]["approach_contact"],
        "gated_contact": summary["gated"]["approach_contact"],
        "baseline_tilt": summary["baseline"]["box_tilt_deg"],
        "gated_tilt": summary["gated"]["box_tilt_deg"],
        "leveled": summary["gated"]["leveled"],
        "final_level_dz": summary["gated"]["final_level_dz"],
        "output": str(out_root),
    }, indent=2))


if __name__ == "__main__":
    main()

