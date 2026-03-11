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


def quat_from_roll(roll):
    cr = float(np.cos(roll * 0.5))
    sr = float(np.sin(roll * 0.5))
    return [cr, sr, 0.0, 0.0]


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
LEVEL_RECOVERY_ROLL_STEP_RAD = np.deg2rad(2.0)
LEVEL_RECOVERY_MAX_ROLL_STEPS = 8
LEVEL_RECOVERY_FINE_ROLL_STEP_RAD = np.deg2rad(1.0)
LEVEL_RECOVERY_FINE_ROLL_STEPS = 2
REANCHOR_XY_TOL = 0.012
DESCENT_MAX_RISE = 0.0015
PRE_CLOSE_POS_TOL = 0.012
PRE_CLOSE_MID_XY_TOL = 0.018
PRE_CLOSE_MAX_ABOVE_TOP = 0.012


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
    jaw_box_cfg = load_jaw_box_config(xml)
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

    def measure_jaw_dz():
        z_fixed_link = float(to_numpy(fixed_jaw.get_pos())[2])
        z_moving_link = float(to_numpy(moving_jaw.get_pos())[2])
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
        z_fixed_surface = float(fixed_inner_surface[2])
        z_moving_surface = float(moving_inner_surface[2])
        mid_inner_surface = 0.5 * (fixed_inner_surface + moving_inner_surface)
        return {
            "dz_link_origin": float(z_moving_link - z_fixed_link),
            "dz_inner_surface": float(z_moving_surface - z_fixed_surface),
            "z_fixed_link_origin": z_fixed_link,
            "z_moving_link_origin": z_moving_link,
            "z_fixed_inner_surface": z_fixed_surface,
            "z_moving_inner_surface": z_moving_surface,
            "mid_inner_surface": np.array(mid_inner_surface, dtype=np.float64),
        }

    def measure_dz_for_qdeg(q_deg, settle_steps=3):
        q_rad = np.deg2rad(np.array(q_deg, dtype=np.float32))
        so101.set_qpos(q_rad)
        so101.control_dofs_position(q_rad, dof_idx)
        for _ in range(settle_steps):
            scene.step()
        # Primary gating metric: inner-surface delta-z.
        return float(measure_jaw_dz()["dz_inner_surface"])

    cube_top_z = float(cube_init[2] + CUBE_SIZE[2] * 0.5)

    def evaluate_qdeg_against_target(q_deg, target_pos, settle_steps=3):
        q_rad = np.deg2rad(np.array(q_deg, dtype=np.float32))
        so101.set_qpos(q_rad)
        so101.control_dofs_position(q_rad, dof_idx)
        for _ in range(settle_steps):
            scene.step()
        jaw = measure_jaw_dz()
        gc = np.array(to_numpy(ee.get_pos()), dtype=np.float64)
        target = np.array(target_pos, dtype=np.float64)
        mid = np.array(jaw["mid_inner_surface"], dtype=np.float64)
        gc_pos_error = float(np.linalg.norm(gc - target))
        gc_xy_drift = float(np.linalg.norm(gc[:2] - target[:2]))
        mid_pos_error = float(np.linalg.norm(mid - target))
        mid_xy_drift = float(np.linalg.norm(mid[:2] - target[:2]))
        return {
            "dz_inner_surface": float(jaw["dz_inner_surface"]),
            "dz_link_origin": float(jaw["dz_link_origin"]),
            "mid_surface_z": float(jaw["mid_inner_surface"][2]),
            "mid_surface_pos_error": mid_pos_error,
            "mid_surface_xy_drift": mid_xy_drift,
            "gc_pos_error": gc_pos_error,
            "gc_xy_drift": gc_xy_drift,
            "mid_surface_to_cube_top": float(jaw["mid_inner_surface"][2] - cube_top_z),
        }

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
    recovery_success_count = 0
    recovery_fail_count = 0
    recovery_events = []
    reanchor_reject_count = 0
    reanchor_reject_events = []
    q_pre_eval = evaluate_qdeg_against_target(q_pre, pos_pre, settle_steps=3)
    prev_mid_surface_z = float(q_pre_eval["mid_surface_z"])
    prev_wp_deg = np.array(q_pre, dtype=np.float64)
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
            dz_wp = measure_dz_for_qdeg(wp)
            if abs(dz_wp) > LEVEL_TOLERANCE:
                best = {
                    "wp": wp,
                    "dz": float(dz_wp),
                    "roll_delta_rad": 0.0,
                    "quat": quat_ref,
                }
                recovered = False
                recovered_choice = None
                if quat_ref is not None:
                    for step in range(1, LEVEL_RECOVERY_MAX_ROLL_STEPS + 1):
                        for sign in (1.0, -1.0):
                            roll_delta = float(sign * step * LEVEL_RECOVERY_ROLL_STEP_RAD)
                            q_try = quat_multiply(
                                quat_from_roll(roll_delta), quat_ref
                            )
                            wp_try = solve_ik_seeded(
                                pos,
                                args.gripper_open,
                                prev_rad,
                                quat_target=q_try,
                            )
                            dz_try = measure_dz_for_qdeg(wp_try)
                            if abs(dz_try) < abs(best["dz"]):
                                best = {
                                    "wp": wp_try,
                                    "dz": float(dz_try),
                                    "roll_delta_rad": float(roll_delta),
                                    "quat": q_try,
                                }
                            if abs(dz_try) <= LEVEL_TOLERANCE:
                                recovered_choice = {
                                    "wp": wp_try,
                                    "dz": float(dz_try),
                                    "roll_delta_rad": float(roll_delta),
                                    "quat": q_try,
                                }
                                recovered = True
                                break
                        if recovered:
                            break
                if recovered:
                    # Fine search around coarse feasible roll with 1-degree resolution.
                    best_feasible = dict(recovered_choice)
                    base_delta = float(recovered_choice["roll_delta_rad"])
                    for step in range(1, LEVEL_RECOVERY_FINE_ROLL_STEPS + 1):
                        for sign in (1.0, -1.0):
                            roll_delta_f = float(
                                base_delta + sign * step * LEVEL_RECOVERY_FINE_ROLL_STEP_RAD
                            )
                            q_try_f = quat_multiply(
                                quat_from_roll(roll_delta_f), quat_ref
                            )
                            wp_try_f = solve_ik_seeded(
                                pos,
                                args.gripper_open,
                                prev_rad,
                                quat_target=q_try_f,
                            )
                            dz_try_f = measure_dz_for_qdeg(wp_try_f)
                            if (
                                abs(dz_try_f) <= LEVEL_TOLERANCE
                                and abs(dz_try_f) < abs(best_feasible["dz"])
                            ):
                                best_feasible = {
                                    "wp": wp_try_f,
                                    "dz": float(dz_try_f),
                                    "roll_delta_rad": float(roll_delta_f),
                                    "quat": q_try_f,
                                }
                    wp = best_feasible["wp"]
                    dz_wp = float(best_feasible["dz"])
                    quat_ref = best_feasible["quat"]
                    recovery_success_count += 1
                    recovery_events.append(
                        {
                            "wp_idx": int(i),
                            "recovered": True,
                            "dz_after": float(dz_wp),
                            "roll_delta_deg": float(np.degrees(best_feasible["roll_delta_rad"])),
                        }
                    )
                else:
                    recovery_fail_count += 1
                    recovery_events.append(
                        {
                            "wp_idx": int(i),
                            "recovered": False,
                            "dz_before": float(dz_wp),
                            "best_dz": float(best["dz"]),
                            "best_roll_delta_deg": float(np.degrees(best["roll_delta_rad"])),
                        }
                    )
                    # No safety fallback: keep descending using the best recovered pose
                    # even if it does not meet the dz threshold.
                    wp = best["wp"]
                    dz_wp = float(best["dz"])
                    quat_ref = best["quat"]
                    gate_blocked = True
                    gate_block_wp = int(i)
                    gate_block_dz = float(best["dz"])
            # Re-anchor to current waypoint target with the recovered quat branch.
            anchor_quat = quat_ref if use_ref_quat else None
            wp_anchor = solve_ik_seeded(
                pos,
                args.gripper_open,
                prev_rad,
                quat_target=anchor_quat,
            )
            anchor_eval = evaluate_qdeg_against_target(wp_anchor, pos, settle_steps=3)
            rise_reject = anchor_eval["mid_surface_z"] > (prev_mid_surface_z + DESCENT_MAX_RISE)
            xy_reject = anchor_eval["mid_surface_xy_drift"] > REANCHOR_XY_TOL
            if rise_reject or xy_reject:
                reanchor_reject_count += 1
                reanchor_reject_events.append(
                    {
                        "wp_idx": int(i),
                        "rise_reject": bool(rise_reject),
                        "xy_reject": bool(xy_reject),
                        "mid_surface_z": float(anchor_eval["mid_surface_z"]),
                        "prev_mid_surface_z": float(prev_mid_surface_z),
                        "mid_surface_xy_drift": float(anchor_eval["mid_surface_xy_drift"]),
                    }
                )
                # Hard reject: keep previous accepted waypoint instead of drifting branch.
                wp = prev_wp_deg.copy()
            else:
                wp = wp_anchor
                prev_mid_surface_z = float(anchor_eval["mid_surface_z"])
                prev_wp_deg = np.array(wp, dtype=np.float64)
        descent_wps.append(wp)
        prev_rad = np.deg2rad(np.array(wp, dtype=np.float32))
    if not descent_wps:
        raise RuntimeError("No descent waypoint generated; check IK/gate settings")
    descent_wps_generated = int(len(descent_wps))
    q_approach = descent_wps[-1]
    pre_close_target = cube_init + off + np.array([0.0, 0.0, args.approach_z], dtype=np.float64)
    pre_close_replan = {
        "enabled": bool(args.quat_mode == "pregrasp_flatten_yaw"),
        "jaw_parallel_ok": None,
        "mid_surface_xy_ok": None,
        "mid_surface_height_ok": None,
        "grasp_center_error_ok": None,
        "pre_close_gate_pass": None,
        "gc_pos_error": None,
        "mid_surface_pos_error": None,
        "mid_surface_xy_drift": None,
        "dz_inner_surface": None,
        "mid_surface_to_cube_top": None,
    }
    if args.quat_mode == "pregrasp_flatten_yaw" and quat_ref is not None:
        q_approach = solve_ik_seeded(
            pre_close_target,
            args.gripper_open,
            prev_rad,
            quat_target=quat_ref,
        )
        pre_close_eval = evaluate_qdeg_against_target(q_approach, pre_close_target, settle_steps=5)
        pre_close_replan.update(
            {
                "gc_pos_error": float(pre_close_eval["gc_pos_error"]),
                "mid_surface_pos_error": float(pre_close_eval["mid_surface_pos_error"]),
                "mid_surface_xy_drift": float(pre_close_eval["mid_surface_xy_drift"]),
                "dz_inner_surface": float(pre_close_eval["dz_inner_surface"]),
                "mid_surface_to_cube_top": float(pre_close_eval["mid_surface_to_cube_top"]),
                "jaw_parallel_ok": bool(abs(pre_close_eval["dz_inner_surface"]) <= LEVEL_TOLERANCE),
                "grasp_center_error_ok": bool(pre_close_eval["gc_pos_error"] <= PRE_CLOSE_POS_TOL),
                "mid_surface_xy_ok": bool(pre_close_eval["mid_surface_xy_drift"] <= PRE_CLOSE_MID_XY_TOL),
                "mid_surface_height_ok": bool(pre_close_eval["mid_surface_to_cube_top"] <= PRE_CLOSE_MAX_ABOVE_TOP),
            }
        )
        pre_close_replan["pre_close_gate_pass"] = bool(
            pre_close_replan["jaw_parallel_ok"]
            and pre_close_replan["mid_surface_xy_ok"]
            and pre_close_replan["mid_surface_height_ok"]
        )
        prev_rad = np.deg2rad(np.array(q_approach, dtype=np.float32))

    q_close = q_approach.copy()
    close_skipped = bool(
        args.quat_mode == "pregrasp_flatten_yaw"
        and pre_close_replan["enabled"]
        and not pre_close_replan["pre_close_gate_pass"]
    )
    if so101.n_dofs >= 6 and not close_skipped:
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
    dz0 = measure_jaw_dz()
    dz_rows.append(
        {
            "frame_idx": -1,
            "phase": "initial_after_reset",
            "dz_jaw": float(dz0["dz_inner_surface"]),
            "dz_inner_surface": float(dz0["dz_inner_surface"]),
            "dz_link_origin": float(dz0["dz_link_origin"]),
        }
    )
    frame_buffer = []
    keep_approach = [i for i, p in enumerate(phases) if p == "approach"]
    keep_approach_tail = set(keep_approach[-EXPORT_APPROACH_TAIL :])
    if args.quat_mode == "pregrasp_flatten_yaw":
        # For gate-leveling runs, keep full pre_grasp_level + approach + close/hold/lift.
        keep_level = {i for i, p in enumerate(phases) if p == "pre_grasp_level"}
        keep_approach_full = set(keep_approach)
        keep_close_and_lift = {
            i
            for i, p in enumerate(phases)
            if p in {"approach_hold", "close", "close_hold", "lift"}
        }
        keep = keep_level | keep_approach_full | keep_close_and_lift
    else:
        # Baseline run: only keep the final approach tail.
        keep = keep_approach_tail
    for i, q_deg in enumerate(traj):
        so101.control_dofs_position(
            np.deg2rad(np.array(q_deg, dtype=np.float32)), dof_idx
        )
        scene.step()
        dz = measure_jaw_dz()
        dz_rows.append(
            {
                "frame_idx": int(i),
                "phase": phases[i],
                "dz_jaw": float(dz["dz_inner_surface"]),
                "dz_inner_surface": float(dz["dz_inner_surface"]),
                "dz_link_origin": float(dz["dz_link_origin"]),
            }
        )
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
        "jaw_reference_for_gating": "inner_surface_from_fixed_and_moving_jaw_box",
        "quat_start_wp": int(QUAT_START_WP),
        "level_tolerance": float(LEVEL_TOLERANCE),
        "level_hold_steps": int(LEVEL_HOLD_STEPS),
        "gate_blocked": bool(gate_blocked),
        "gate_block_wp": gate_block_wp,
        "gate_block_dz": gate_block_dz,
        "level_recovery": {
            "roll_step_deg": float(np.degrees(LEVEL_RECOVERY_ROLL_STEP_RAD)),
            "max_roll_steps": int(LEVEL_RECOVERY_MAX_ROLL_STEPS),
            "fine_roll_step_deg": float(np.degrees(LEVEL_RECOVERY_FINE_ROLL_STEP_RAD)),
            "fine_roll_steps": int(LEVEL_RECOVERY_FINE_ROLL_STEPS),
            "success_count": int(recovery_success_count),
            "fail_count": int(recovery_fail_count),
            "events": recovery_events,
        },
        "reanchor_checks": {
            "xy_tol": float(REANCHOR_XY_TOL),
            "descent_max_rise": float(DESCENT_MAX_RISE),
            "reject_count": int(reanchor_reject_count),
            "events": reanchor_reject_events,
        },
        "pre_close_replan": pre_close_replan,
        "close_skipped_by_pre_close_gate": bool(close_skipped),
        "descent_wps_planned": int(n_descent_wps),
        "descent_wps_used": int(descent_wps_generated),
        "descent_wps_executed": int(len(descent_wps)),
        "descent_wps_padded": 0,
        "gripper_open": float(args.gripper_open),
        "gripper_close": float(args.gripper_close),
        "cube_shift_norm": float(np.linalg.norm(cube_shift)),
        "cube_shift_xyz": [float(v) for v in cube_shift.tolist()],
        "cube_tilt_deg": float(box_tilt_deg(to_numpy(cube.get_quat()))),
        "camera_side_pose": {"pos": list(side_pos), "lookat": list(side_lookat)},
        "export_frames": {
            "approach_tail": int(EXPORT_APPROACH_TAIL),
            "pre_grasp_level_all_when_flatten_yaw": bool(
                args.quat_mode == "pregrasp_flatten_yaw"
            ),
            "approach_all_when_flatten_yaw": bool(
                args.quat_mode == "pregrasp_flatten_yaw"
            ),
            "close_close_hold_lift_all_when_flatten_yaw": bool(
                args.quat_mode == "pregrasp_flatten_yaw"
            ),
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
