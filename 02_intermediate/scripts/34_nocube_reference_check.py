"""
No-cube reference check for grasp-center vs jaw-midpoint bias.

Purpose:
- Validate whether grasp_center has a systematic Z bias relative to jaw midpoint.
- Run a light trajectory without cube/contact and log:
  z_fixed, z_moving, z_grasp_center,
  mid_jaw=(z_fixed+z_moving)/2,
  error=z_grasp_center-mid_jaw.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET
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


def parse_vec(s):
    return np.array([float(v) for v in s.split()], dtype=np.float64)


def normalize(v):
    v = np.array(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def transform_point(link_pos, link_quat, local_pos):
    return np.array(link_pos, dtype=np.float64) + quat_to_rotmat(link_quat) @ np.array(
        local_pos, dtype=np.float64
    )


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


def load_jaw_box_config(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("worldbody not found in XML")
    jaw_boxes = {}

    def walk_body(body):
        body_name = body.attrib.get("name")
        for geom in body.findall("geom"):
            geom_name = geom.attrib.get("name")
            if geom_name in {"fixed_jaw_box", "moving_jaw_box"}:
                jaw_boxes[geom_name] = {
                    "body_name": body_name,
                    "pos": parse_vec(geom.attrib["pos"]),
                    "size": parse_vec(geom.attrib["size"]),
                }
        for child in body.findall("body"):
            walk_body(child)

    for body in worldbody.findall("body"):
        walk_body(body)

    missing = {"fixed_jaw_box", "moving_jaw_box"} - set(jaw_boxes.keys())
    if missing:
        raise RuntimeError(f"jaw box geom not found in XML: {sorted(missing)}")
    return jaw_boxes


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
    return (q / (np.linalg.norm(q) + 1e-9)).tolist()


def quat_from_yaw(yaw):
    cy = float(np.cos(yaw * 0.5))
    sy = float(np.sin(yaw * 0.5))
    return [cy, 0.0, 0.0, sy]


def yaw_from_quat_x_axis(q):
    r = quat_to_rotmat(np.array(q, dtype=np.float64))
    x_axis_world = r @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return float(np.arctan2(x_axis_world[1], x_axis_world[0]))


def lerp(a, b, n):
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    return [a + (b - a) * (i + 1) / max(n, 1) for i in range(n)]


HOME_DEG = np.array([0.0, -30.0, 90.0, -60.0, 0.0, 0.0], dtype=np.float32)
KP = np.array([500.0, 500.0, 400.0, 400.0, 300.0, 200.0], dtype=np.float32)
KV = np.array([50.0, 50.0, 40.0, 40.0, 30.0, 20.0], dtype=np.float32)


def stat(vals):
    arr = np.array(vals, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main() -> None:
    ensure_display()
    ap = argparse.ArgumentParser(description="No-cube reference bias check")
    ap.add_argument("--exp-id", default="nocube_reference_check")
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
    ap.add_argument("--cube-x", type=float, default=0.16, help="Virtual target x")
    ap.add_argument("--cube-y", type=float, default=0.0, help="Virtual target y")
    ap.add_argument("--cube-z", type=float, default=0.015, help="Virtual target z")
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
    )
    args = ap.parse_args()

    xml = find_so101_xml(args.xml)
    if xml is None:
        raise RuntimeError("SO101 xml not found")

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
    so101 = scene.add_entity(gs.morphs.MJCF(file=str(xml), pos=(0.0, 0.0, 0.0)))
    scene.build()

    dof_idx = np.arange(so101.n_dofs)
    so101.set_dofs_kp(KP[: so101.n_dofs], dof_idx)
    so101.set_dofs_kv(KV[: so101.n_dofs], dof_idx)
    home_deg = HOME_DEG[: so101.n_dofs]
    home_rad = np.deg2rad(home_deg)

    ee = so101.get_link("grasp_center")
    fixed_jaw = so101.get_link("gripper")
    moving_jaw = so101.get_link("moving_jaw_so101_v1")
    jaw_box_cfg = load_jaw_box_config(xml)

    target = np.array([args.cube_x, args.cube_y, args.cube_z], dtype=np.float64)
    off = np.array(
        [args.grasp_offset_x, args.grasp_offset_y, args.grasp_offset_z], dtype=np.float64
    )

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

    def reset_scene():
        so101.set_qpos(home_rad)
        so101.control_dofs_position(home_rad, dof_idx)
        so101.zero_all_dofs_velocity()
        for _ in range(args.settle_steps):
            scene.step()

    def get_jaw_box_world(link, cfg):
        link_pos = to_numpy(link.get_pos())
        link_quat = to_numpy(link.get_quat())
        center_world = transform_point(link_pos, link_quat, cfg["pos"])
        rot = quat_to_rotmat(link_quat)
        thickness_axis_local = np.zeros(3, dtype=np.float64)
        thickness_axis_local[int(np.argmin(cfg["size"]))] = 1.0
        thickness_axis_world = normalize(rot @ thickness_axis_local)
        return {
            "center_world": center_world,
            "thickness_axis_world": thickness_axis_world,
            "half_thickness": float(np.min(cfg["size"])),
        }

    def compute_jaw_surface_midpoint_world():
        fixed_box = get_jaw_box_world(fixed_jaw, jaw_box_cfg["fixed_jaw_box"])
        moving_box = get_jaw_box_world(moving_jaw, jaw_box_cfg["moving_jaw_box"])
        fixed_axis = fixed_box["thickness_axis_world"]
        moving_axis = moving_box["thickness_axis_world"]
        fixed_to_moving = moving_box["center_world"] - fixed_box["center_world"]
        moving_to_fixed = fixed_box["center_world"] - moving_box["center_world"]
        fixed_inward = fixed_axis * np.sign(np.dot(fixed_to_moving, fixed_axis) or 1.0)
        moving_inward = moving_axis * np.sign(np.dot(moving_to_fixed, moving_axis) or 1.0)
        fixed_inner_surface = (
            fixed_box["center_world"] + fixed_inward * fixed_box["half_thickness"]
        )
        moving_inner_surface = (
            moving_box["center_world"] + moving_inward * moving_box["half_thickness"]
        )
        jaw_midpoint = 0.5 * (fixed_inner_surface + moving_inner_surface)
        return fixed_inner_surface, moving_inner_surface, jaw_midpoint

    pos_pre = target + off + np.array([0.0, 0.0, 0.10], dtype=np.float64)
    q_pre = solve_ik_seeded(pos_pre, args.gripper_open, home_rad, quat_target=None)
    prev_rad = np.deg2rad(np.array(q_pre, dtype=np.float32))

    quat_ref = None
    quat_ref_source = "disabled"
    if args.quat_mode == "pregrasp_flatten_yaw":
        q_pre_rad = np.deg2rad(np.array(q_pre, dtype=np.float32))
        so101.set_qpos(q_pre_rad)
        so101.control_dofs_position(q_pre_rad, dof_idx)
        for _ in range(5):
            scene.step()
        quat_pre = [float(v) for v in to_numpy(ee.get_quat()).tolist()]
        yaw = yaw_from_quat_x_axis(quat_pre)
        quat_ref = quat_multiply(quat_from_yaw(yaw), [0.0, 1.0, 0.0, 0.0])
        quat_ref_source = "auto_pre_grasp_flatten_yaw"
        reset_scene()

    n_descent_wps = 6
    descent_wps = []
    for i in range(n_descent_wps):
        frac = (i + 1) / n_descent_wps
        z = 0.10 + (args.approach_z - 0.10) * frac
        pos = target + off + np.array([0.0, 0.0, z], dtype=np.float64)
        use_ref_quat = args.quat_mode == "pregrasp_flatten_yaw" and i >= 2
        wp = solve_ik_seeded(
            pos,
            args.gripper_open,
            prev_rad,
            quat_target=quat_ref if use_ref_quat else None,
        )
        descent_wps.append(wp)
        prev_rad = np.deg2rad(np.array(wp, dtype=np.float32))

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
        pos = target + off + np.array([0.0, 0.0, z], dtype=np.float64)
        wp = solve_ik_seeded(
            pos,
            args.gripper_close,
            prev_rad,
            quat_target=quat_ref if args.quat_mode == "pregrasp_flatten_yaw" else None,
        )
        lift_wps.append(wp)
        prev_rad = np.deg2rad(np.array(wp, dtype=np.float32))

    traj = []
    phases = []
    n_move = max(15, args.trial_steps // 8)
    traj += lerp(home_deg.copy(), q_pre, n_move)
    phases += ["move_pre"] * n_move
    traj += [q_pre.copy() for _ in range(8)]
    phases += ["pre_grasp_level"] * 8
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

    reset_scene()
    rows = []
    for i, q_deg in enumerate(traj):
        so101.control_dofs_position(np.deg2rad(np.array(q_deg, dtype=np.float32)), dof_idx)
        scene.step()
        z_fixed = float(to_numpy(fixed_jaw.get_pos())[2])
        z_moving = float(to_numpy(moving_jaw.get_pos())[2])
        z_gc = float(to_numpy(ee.get_pos())[2])
        mid_link_origin = 0.5 * (z_fixed + z_moving)
        err_link_origin = float(z_gc - mid_link_origin)

        fixed_inner_surface, moving_inner_surface, jaw_mid_surface = (
            compute_jaw_surface_midpoint_world()
        )
        z_fixed_surface = float(fixed_inner_surface[2])
        z_moving_surface = float(moving_inner_surface[2])
        z_mid_surface = float(jaw_mid_surface[2])
        err_surface = float(z_gc - z_mid_surface)

        rows.append(
            {
                "frame_idx": int(i),
                "phase": phases[i],
                "z_fixed_link_origin": z_fixed,
                "z_moving_link_origin": z_moving,
                "z_grasp_center": z_gc,
                "z_fixed_inner_surface": z_fixed_surface,
                "z_moving_inner_surface": z_moving_surface,
                "z_jaw_mid_link_origin": float(mid_link_origin),
                "z_jaw_mid_inner_surface": z_mid_surface,
                "delta_z_link_origin": float(z_moving - z_fixed),
                "delta_z_inner_surface": float(z_moving_surface - z_fixed_surface),
                "error_center_minus_midjaw_link_origin": err_link_origin,
                "error_center_minus_midjaw_inner_surface": err_surface,
            }
        )

    phase_names = sorted(set(r["phase"] for r in rows))
    phase_stats = {}
    for ph in phase_names:
        sub = [r for r in rows if r["phase"] == ph]
        phase_stats[ph] = {
            "delta_z_link_origin": stat([r["delta_z_link_origin"] for r in sub]),
            "delta_z_inner_surface": stat([r["delta_z_inner_surface"] for r in sub]),
            "error_center_minus_midjaw_link_origin": stat(
                [r["error_center_minus_midjaw_link_origin"] for r in sub]
            ),
            "error_center_minus_midjaw_inner_surface": stat(
                [r["error_center_minus_midjaw_inner_surface"] for r in sub]
            ),
        }

    summary = {
        "exp_id": args.exp_id,
        "xml_path": str(xml),
        "quat_mode": args.quat_mode,
        "quat_ref_source": quat_ref_source,
        "target_xyz": [float(v) for v in target.tolist()],
        "offset_xyz": [float(v) for v in off.tolist()],
        "jaw_reference": "inner_surface_from_fixed_and_moving_jaw_box",
        "global_stats": {
            "delta_z_link_origin": stat([r["delta_z_link_origin"] for r in rows]),
            "delta_z_inner_surface": stat([r["delta_z_inner_surface"] for r in rows]),
            "error_center_minus_midjaw_link_origin": stat(
                [r["error_center_minus_midjaw_link_origin"] for r in rows]
            ),
            "error_center_minus_midjaw_inner_surface": stat(
                [r["error_center_minus_midjaw_inner_surface"] for r in rows]
            ),
            "z_fixed_link_origin": stat([r["z_fixed_link_origin"] for r in rows]),
            "z_moving_link_origin": stat([r["z_moving_link_origin"] for r in rows]),
            "z_fixed_inner_surface": stat([r["z_fixed_inner_surface"] for r in rows]),
            "z_moving_inner_surface": stat([r["z_moving_inner_surface"] for r in rows]),
            "z_grasp_center": stat([r["z_grasp_center"] for r in rows]),
        },
        "phase_stats": phase_stats,
        "samples": rows,
    }

    out_root = Path(args.save) / args.exp_id
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "nocube_reference_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
