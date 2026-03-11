"""
Minimal grasp script extracted from 3_grasp_experiment.py.

Goal:
- Keep only one clean grasp pipeline: reset -> approach -> close -> lift.
- Make it easy to test whether failures come from core geometry/control
  or from extra experiment logic in 3_grasp_experiment.py.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np


def ensure_display() -> None:
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


def save_rgb_png(arr, path: Path) -> None:
    try:
        from PIL import Image

        Image.fromarray(arr).save(path)
    except ImportError:
        import imageio.v2 as imageio

        imageio.imwrite(path, arr)


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    q = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )
    n = np.linalg.norm(q)
    return (q / (n + 1e-9)).tolist()


def quat_from_yaw(yaw):
    cy = float(np.cos(yaw * 0.5))
    sy = float(np.sin(yaw * 0.5))
    return [cy, 0.0, 0.0, sy]


def yaw_from_quat_x_axis(q):
    # Estimate yaw by projecting ee local X axis onto world XY.
    r = quat_to_rotmat(np.array(q, dtype=np.float64))
    x_axis_world = r @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return float(np.arctan2(x_axis_world[1], x_axis_world[0]))


def find_so101_xml(user_xml: str | None) -> Path | None:
    if user_xml:
        p = Path(user_xml)
        return p if p.exists() else None
    here = Path(__file__).resolve().parent
    candidates = [
        here / "assets" / "so101_new_calib_v4.xml",
        here / "assets" / "so101_new_calib.xml",
        Path("02_intermediate/scripts/assets/so101_new_calib_v4.xml"),
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


def box_tilt_deg(cube_quat) -> float:
    r = quat_to_rotmat(cube_quat)
    z_axis = r @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cosv = abs(
        float(
            np.clip(
                np.dot(
                    z_axis / (np.linalg.norm(z_axis) + 1e-9), np.array([0.0, 0.0, 1.0])
                ),
                -1.0,
                1.0,
            )
        )
    )
    return float(np.degrees(np.arccos(cosv)))


def lerp(a, b, n):
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    return [a + (b - a) * (i + 1) / max(n, 1) for i in range(n)]


HOME_DEG = np.array([0.0, -30.0, 90.0, -60.0, 0.0, 0.0], dtype=np.float32)
KP = np.array([500.0, 500.0, 400.0, 400.0, 300.0, 200.0], dtype=np.float32)
KV = np.array([50.0, 50.0, 40.0, 40.0, 30.0, 20.0], dtype=np.float32)
CUBE_SIZE = (0.03, 0.03, 0.03)
QUAT_START_WP = 2
LEVEL_TOLERANCE = 0.004
LEVEL_HOLD_STEPS = 8
EXPORT_APPROACH_TAIL = 10
EXPORT_LEVEL_TAIL = 8


def phase_stats(rows):
    if not rows:
        return {}
    by_phase = {}
    for r in rows:
        by_phase.setdefault(r["phase"], []).append(float(r["dz_jaw"]))
    out = {}
    for ph, vals in by_phase.items():
        arr = np.array(vals, dtype=np.float64)
        out[ph] = {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return out


def main() -> None:
    ensure_display()
    ap = argparse.ArgumentParser(description="SO-101 minimal light grasp")
    ap.add_argument("--exp-id", default="grasp_light")
    ap.add_argument("--xml", default=None)
    ap.add_argument("--save", default="/output")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--sim-substeps", type=int, default=4)
    ap.add_argument("--sim-dt", type=float, default=None)
    ap.add_argument("--trial-steps", type=int, default=90)
    ap.add_argument("--approach-hold-steps", type=int, default=1)
    ap.add_argument("--close-hold-steps", type=int, default=12)
    ap.add_argument("--settle-steps", type=int, default=30)
    ap.add_argument("--cube-x", type=float, default=0.16)
    ap.add_argument("--cube-y", type=float, default=0.0)
    ap.add_argument("--cube-z", type=float, default=0.015)
    ap.add_argument("--cube-friction", type=float, default=1.5)
    ap.add_argument("--gripper-open", type=float, default=20.0)
    ap.add_argument("--gripper-close", type=float, default=2.0)
    ap.add_argument("--grasp-offset-x", type=float, default=0.0)
    ap.add_argument("--grasp-offset-y", type=float, default=0.0)
    ap.add_argument("--grasp-offset-z", type=float, default=-0.01)
    ap.add_argument("--approach-z", type=float, default=0.012)
    ap.add_argument(
        "--quat-mode",
        choices=["none", "pregrasp_flatten_yaw"],
        default="none",
        help="none: default IK; pregrasp_flatten_yaw: enforce flattened-yaw quat on late descent/approach",
    )
    args = ap.parse_args()

    xml = find_so101_xml(args.xml)
    if xml is None:
        raise RuntimeError("SO101 xml not found")

    import torch
    import genesis as gs

    gs.init(backend=(gs.cpu if args.cpu else gs.gpu), logging_level="warning")
    sim_dt = args.sim_dt if args.sim_dt is not None else 1.0 / args.fps
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=sim_dt, substeps=args.sim_substeps),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True, enable_joint_limit=True, box_box_detection=True
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    cube = scene.add_entity(
        morph=gs.morphs.Box(
            size=CUBE_SIZE, pos=(args.cube_x, args.cube_y, args.cube_z)
        ),
        material=gs.materials.Rigid(friction=args.cube_friction),
        surface=gs.surfaces.Default(color=(1.0, 0.3, 0.3, 1.0)),
    )
    so101 = scene.add_entity(gs.morphs.MJCF(file=str(xml), pos=(0.0, 0.0, 0.0)))
    cam_top = scene.add_camera(
        res=(640, 480),
        pos=(0.42, 0.34, 0.26),
        lookat=(0.15, 0.0, 0.08),
        fov=38,
        GUI=False,
    )
    # Side camera: keep axis-aligned view but move farther so full arm fits in FOV.
    side_pos = (float(args.cube_x), float(args.cube_y - 0.32), float(args.cube_z + 0.09))
    side_lookat = (float(args.cube_x), float(args.cube_y), float(args.cube_z + 0.03))
    cam_side = scene.add_camera(
        res=(640, 480), pos=side_pos, lookat=side_lookat, fov=50, GUI=False
    )
    scene.build()

    dof_idx = np.arange(so101.n_dofs)
    so101.set_dofs_kp(KP[: so101.n_dofs], dof_idx)
    so101.set_dofs_kv(KV[: so101.n_dofs], dof_idx)
    home_deg = HOME_DEG[: so101.n_dofs]
    home_rad = np.deg2rad(home_deg)
    ee = so101.get_link("grasp_center")
    fixed_jaw = so101.get_link("gripper")
    moving_jaw = so101.get_link("moving_jaw_so101_v1")
    cube_init = np.array([args.cube_x, args.cube_y, args.cube_z], dtype=np.float64)
    off = np.array(
        [args.grasp_offset_x, args.grasp_offset_y, args.grasp_offset_z],
        dtype=np.float64,
    )

    def reset_scene():
        so101.set_qpos(home_rad)
        so101.control_dofs_position(home_rad, dof_idx)
        so101.zero_all_dofs_velocity()
        cube.set_pos(
            torch.tensor(cube_init, dtype=torch.float32, device=gs.device).unsqueeze(0)
        )
        cube.set_quat(
            torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=gs.device).unsqueeze(
                0
            )
        )
        cube.zero_all_dofs_velocity()
        for _ in range(args.settle_steps):
            scene.step()

    def solve_ik_seeded(pos, grip_deg, seed_rad, quat_target=None):
        q = to_numpy(
            so101.inverse_kinematics(
                link=ee,
                pos=np.array(pos, dtype=np.float32),
                quat=quat_target,
                init_qpos=seed_rad,
                max_solver_iters=50,
                damping=0.02,
            )
        )
        q_deg = np.rad2deg(q)
        if so101.n_dofs >= 6:
            q_deg[5] = grip_deg
        return q_deg

    # trajectory (same skeleton as 03/12)
    pos_pre = cube_init + off + np.array([0.0, 0.0, 0.10], dtype=np.float64)
    q_pre = solve_ik_seeded(pos_pre, args.gripper_open, home_rad, quat_target=None)
    prev_rad = np.deg2rad(np.array(q_pre, dtype=np.float32))
    quat_ref = None
    quat_ref_source = "disabled"
    if args.quat_mode == "pregrasp_flatten_yaw":
        # Build leveling-oriented quat_ref from pre_grasp reachable pose.
        q_pre_rad = np.deg2rad(np.array(q_pre, dtype=np.float32))
        so101.set_qpos(q_pre_rad)
        so101.control_dofs_position(q_pre_rad, dof_idx)
        for _ in range(5):
            scene.step()
        quat_pre = [float(v) for v in to_numpy(ee.get_quat()).tolist()]
        # Keep yaw from current reachable branch, flatten roll/pitch to down-facing base.
        yaw = yaw_from_quat_x_axis(quat_pre)
        q_yaw = quat_from_yaw(yaw)
        q_down = [0.0, 1.0, 0.0, 0.0]
        quat_ref = quat_multiply(q_yaw, q_down)
        quat_ref_source = "auto_pre_grasp_flatten_yaw"
        # Restore home so the actual run still starts from reset_scene().
        so101.set_qpos(home_rad)
        so101.control_dofs_position(home_rad, dof_idx)
        so101.zero_all_dofs_velocity()
        for _ in range(5):
            scene.step()
    n_descent_wps = 6
    descent_wps = []
    gate_blocked = False
    gate_block_wp = None
    gate_block_dz = None
    for i in range(n_descent_wps):
        frac = (i + 1) / n_descent_wps
        z = 0.10 + (args.approach_z - 0.10) * frac
        pos = cube_init + off + np.array([0.0, 0.0, z], dtype=np.float64)
        use_ref_quat = args.quat_mode == "pregrasp_flatten_yaw" and i >= QUAT_START_WP
        wp = solve_ik_seeded(
            pos,
            args.gripper_open,
            prev_rad,
            quat_target=quat_ref if use_ref_quat else None,
        )
        # Gate: from quat-start waypoint onward, only allow deeper descend when dz_jaw is within tolerance.
        if args.quat_mode == "pregrasp_flatten_yaw" and i >= QUAT_START_WP:
            q_wp_rad = np.deg2rad(np.array(wp, dtype=np.float32))
            so101.set_qpos(q_wp_rad)
            so101.control_dofs_position(q_wp_rad, dof_idx)
            for _ in range(3):
                scene.step()
            dz_wp = float(to_numpy(moving_jaw.get_pos())[2] - to_numpy(fixed_jaw.get_pos())[2])
            if abs(dz_wp) > LEVEL_TOLERANCE:
                gate_blocked = True
                gate_block_wp = int(i)
                gate_block_dz = float(dz_wp)
                break
        descent_wps.append(wp)
        prev_rad = np.deg2rad(np.array(wp, dtype=np.float32))
    if not descent_wps:
        raise RuntimeError("No descent waypoint generated; check IK/gate settings")
    q_approach = descent_wps[-1]

    q_close = q_approach.copy()
    if so101.n_dofs >= 6:
        q_close[5] = args.gripper_close

    n_lift_wps = 4
    lift_wps = []
    prev_rad = np.deg2rad(np.array(q_close, dtype=np.float32))
    for i in range(n_lift_wps):
        frac = (i + 1) / n_lift_wps
        z = args.approach_z + (0.15 - args.approach_z) * frac
        pos = cube_init + off + np.array([0.0, 0.0, z], dtype=np.float64)
        wp = solve_ik_seeded(
            pos,
            args.gripper_close,
            prev_rad,
            quat_target=quat_ref if args.quat_mode == "pregrasp_flatten_yaw" else None,
        )
        lift_wps.append(wp)
        prev_rad = np.deg2rad(np.array(wp, dtype=np.float32))

    # dense execution trajectory
    traj = []
    phases = []
    n_move = max(15, args.trial_steps // 8)
    traj += lerp(home_deg.copy(), q_pre, n_move)
    phases += ["move_pre"] * n_move
    if LEVEL_HOLD_STEPS > 0:
        traj += [q_pre.copy() for _ in range(LEVEL_HOLD_STEPS)]
        phases += ["pre_grasp_level"] * LEVEL_HOLD_STEPS
    steps_per_desc = max(3, args.trial_steps // (8 * n_descent_wps))
    prev = q_pre
    for wp in descent_wps:
        seg = lerp(prev, wp, steps_per_desc)
        traj += seg
        phases += ["approach"] * len(seg)
        prev = wp
    traj += [q_approach.copy() for _ in range(args.approach_hold_steps)]
    phases += ["approach_hold"] * args.approach_hold_steps
    n_close = max(8, args.trial_steps // 12)
    traj += lerp(q_approach, q_close, n_close)
    phases += ["close"] * n_close
    traj += [q_close.copy() for _ in range(args.close_hold_steps)]
    phases += ["close_hold"] * args.close_hold_steps
    steps_per_lift = max(5, args.trial_steps // (8 * n_lift_wps))
    prev = q_close
    for wp in lift_wps:
        seg = lerp(prev, wp, steps_per_lift)
        traj += seg
        phases += ["lift"] * len(seg)
        prev = wp

    out_root = Path(args.save) / args.exp_id
    out_root.mkdir(parents=True, exist_ok=True)
    png_dir = out_root / "approach_debug_pngs"

    reset_scene()
    dz_rows = []
    dz0 = float(to_numpy(moving_jaw.get_pos())[2] - to_numpy(fixed_jaw.get_pos())[2])
    dz_rows.append({"frame_idx": -1, "phase": "initial_after_reset", "dz_jaw": dz0})
    frame_buffer = []
    keep_approach = [i for i, p in enumerate(phases) if p == "approach"]
    keep_approach = set(keep_approach[-EXPORT_APPROACH_TAIL :])
    keep_level = [i for i, p in enumerate(phases) if p == "pre_grasp_level"]
    keep_level = set(keep_level[-EXPORT_LEVEL_TAIL :])
    keep = keep_approach | keep_level
    for i, q_deg in enumerate(traj):
        so101.control_dofs_position(
            np.deg2rad(np.array(q_deg, dtype=np.float32)), dof_idx
        )
        scene.step()
        dz = float(to_numpy(moving_jaw.get_pos())[2] - to_numpy(fixed_jaw.get_pos())[2])
        dz_rows.append({"frame_idx": int(i), "phase": phases[i], "dz_jaw": dz})
        if i in keep:
            frame_buffer.append(
                (
                    i,
                    phases[i],
                    np.concatenate(
                        [render_camera(cam_top), render_camera(cam_side)], axis=1
                    ),
                )
            )

    png_dir.mkdir(parents=True, exist_ok=True)
    for i, phase, img in frame_buffer:
        save_rgb_png(img, png_dir / f"f{i:03d}_{phase}.png")

    cube_pos_final = to_numpy(cube.get_pos())
    cube_shift = cube_pos_final - cube_init
    phase_dz = phase_stats(dz_rows)
    approach_rows = [r for r in dz_rows if r["phase"] == "approach"]
    move_rows = [r for r in dz_rows if r["phase"] == "move_pre"]
    dz_analysis = {}
    if move_rows and approach_rows:
        move_start = float(move_rows[0]["dz_jaw"])
        move_end = float(move_rows[-1]["dz_jaw"])
        app_start = float(approach_rows[0]["dz_jaw"])
        app_end = float(approach_rows[-1]["dz_jaw"])
        x = np.arange(len(approach_rows), dtype=np.float64)
        y = np.array([r["dz_jaw"] for r in approach_rows], dtype=np.float64)
        slope = float(np.polyfit(x, y, 1)[0]) if len(approach_rows) >= 2 else 0.0
        dz_analysis = {
            "move_pre_start": move_start,
            "move_pre_end": move_end,
            "approach_start": app_start,
            "approach_end": app_end,
            "delta_move_pre": float(move_end - move_start),
            "delta_approach": float(app_end - app_start),
            "approach_slope_per_step": slope,
            # simple heuristic: |delta_approach| > 1 mm and trend sign matches
            "approach_accumulates": bool(abs(app_end - app_start) > 1e-3 and np.sign(app_end - app_start) == np.sign(slope)),
        }
    summary = {
        "exp_id": args.exp_id,
        "xml_path": str(xml),
        "cube_init": cube_init.tolist(),
        "offset_xyz": off.tolist(),
        "settle_steps": int(args.settle_steps),
        "approach_z": float(args.approach_z),
        "quat_mode": args.quat_mode,
        "quat_ref_source": quat_ref_source,
        "quat_ref": quat_ref,
        "quat_start_wp": int(QUAT_START_WP),
        "level_tolerance": float(LEVEL_TOLERANCE),
        "level_hold_steps": int(LEVEL_HOLD_STEPS),
        "gate_blocked": bool(gate_blocked),
        "gate_block_wp": gate_block_wp,
        "gate_block_dz": gate_block_dz,
        "descent_wps_used": int(len(descent_wps)),
        "gripper_open": float(args.gripper_open),
        "gripper_close": float(args.gripper_close),
        "cube_shift_norm": float(np.linalg.norm(cube_shift)),
        "cube_shift_xyz": [float(v) for v in cube_shift.tolist()],
        "cube_tilt_deg": float(box_tilt_deg(to_numpy(cube.get_quat()))),
        "camera_side_pose": {"pos": list(side_pos), "lookat": list(side_lookat)},
        "export_frames": {
            "approach_tail": int(EXPORT_APPROACH_TAIL),
            "pre_grasp_level_tail": int(EXPORT_LEVEL_TAIL),
        },
        "dz_jaw_phase_stats": phase_dz,
        "dz_jaw_analysis": dz_analysis,
        "dz_jaw_trace": dz_rows,
        "approach_png_dir": str(png_dir),
    }
    (out_root / "light_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
