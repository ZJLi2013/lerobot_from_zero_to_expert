"""
SO-101 grasp with lightweight yaw-snap alignment.

Pipeline: reset -> descent (position-only IK + seed chain) -> yaw snap -> close -> lift
Based on 33_grasp_light.py with all quat_ref / tune-roll / replan stripped.

Yaw snap: after descent, measure jaw closing-axis yaw in XY plane,
then adjust wrist_roll joint to snap to nearest 0/90 degree alignment
with cube faces. This is a pure post-processing step — no IK re-solve needed.
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


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ensure_display() -> None:
    if os.environ.get("DISPLAY"):
        return
    xvfb = subprocess.run(["which", "Xvfb"], capture_output=True)
    if xvfb.returncode != 0:
        return
    proc = subprocess.Popen(
        ["Xvfb", ":99", "-screen", "0", "1280x1024x24", "-ac", "+extension", "GLX"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
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
    return v / n if n > 1e-12 else np.zeros_like(v)


def transform_point(link_pos, link_quat, local_pos):
    return np.array(link_pos, dtype=np.float64) + quat_to_rotmat(link_quat) @ np.array(local_pos, dtype=np.float64)


def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def render_camera(cam):
    rgb, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, normal=False)
    arr = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.array(rgb)
    return arr[0] if arr.ndim == 4 else arr
    return arr.astype(np.uint8)


def save_rgb_png(arr, path: Path) -> None:
    try:
        from PIL import Image
        Image.fromarray(arr).save(path)
    except ImportError:
        import imageio.v2 as imageio
        imageio.imwrite(path, arr)


def box_tilt_deg(cube_quat) -> float:
    r = quat_to_rotmat(cube_quat)
    z_axis = r @ np.array([0.0, 0.0, 1.0])
    cosv = abs(float(np.clip(np.dot(z_axis / (np.linalg.norm(z_axis) + 1e-9),
                                     np.array([0.0, 0.0, 1.0])), -1, 1)))
    return float(np.degrees(np.arccos(cosv)))


def find_so101_xml(user_xml: str | None) -> Path | None:
    if user_xml:
        p = Path(user_xml)
        return p if p.exists() else None
    here = Path(__file__).resolve().parent
    for p in [
        here / "assets" / "so101_new_calib_v4.xml",
        here / "assets" / "so101_new_calib.xml",
        Path("02_intermediate/scripts/assets/so101_new_calib_v4.xml"),
        Path("02_intermediate/scripts/assets/so101_new_calib.xml"),
    ]:
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
    def walk(body):
        for geom in body.findall("geom"):
            gn = geom.attrib.get("name")
            if gn in {"fixed_jaw_box", "moving_jaw_box"}:
                jaw_boxes[gn] = {
                    "body_name": body.attrib.get("name"),
                    "pos": parse_vec(geom.attrib["pos"]),
                    "size": parse_vec(geom.attrib["size"]),
                }
        for child in body.findall("body"):
            walk(child)
    for body in worldbody.findall("body"):
        walk(body)
    missing = {"fixed_jaw_box", "moving_jaw_box"} - set(jaw_boxes)
    if missing:
        raise RuntimeError(f"jaw box geom not found: {sorted(missing)}")
    return jaw_boxes


def lerp(a, b, n):
    a, b = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
    return [a + (b - a) * (i + 1) / max(n, 1) for i in range(n)]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HOME_DEG = np.array([0.0, -30.0, 90.0, -60.0, 0.0, 0.0], dtype=np.float32)
KP = np.array([500.0, 500.0, 400.0, 400.0, 300.0, 200.0], dtype=np.float32)
KV = np.array([50.0, 50.0, 40.0, 40.0, 30.0, 20.0], dtype=np.float32)
CUBE_SIZE = (0.03, 0.03, 0.03)

N_DESCENT = 10
N_CLOSE = 4
N_LIFT_WPS = 4
PRE_HEIGHT = 0.15
WRIST_ROLL_IDX = 4
SNAP_BUFFER_STEPS = 4


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_display()

    ap = argparse.ArgumentParser(description="SO-101 grasp with yaw-snap alignment")
    ap.add_argument("--exp-id", default="grasp_yaw_snap")
    ap.add_argument("--xml", default=None)
    ap.add_argument("--save", default="/output")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--sim-substeps", type=int, default=4)
    ap.add_argument("--settle-steps", type=int, default=30)
    ap.add_argument("--close-hold-steps", type=int, default=12)
    ap.add_argument("--cube-x", type=float, default=0.15)
    ap.add_argument("--cube-y", type=float, default=-0.06)
    ap.add_argument("--cube-z", type=float, default=None,
                    help="Cube center Z. Default: CUBE_SIZE[2]/2")
    ap.add_argument("--cube-friction", type=float, default=1.5)
    ap.add_argument("--gripper-open", type=float, default=25.0)
    ap.add_argument("--gripper-close", type=float, default=2.0)
    ap.add_argument("--grasp-offset-z", type=float, default=0.0)
    ap.add_argument("--approach-z", type=float, default=0.012)
    args = ap.parse_args()

    if args.cube_z is None:
        args.cube_z = CUBE_SIZE[2] / 2.0

    xml = find_so101_xml(args.xml)
    if xml is None:
        raise RuntimeError("SO101 xml not found")

    import torch
    import genesis as gs

    gs.init(backend=(gs.cpu if args.cpu else gs.gpu), logging_level="warning")
    sim_dt = 1.0 / args.fps
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=sim_dt, substeps=args.sim_substeps),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True, enable_joint_limit=True, box_box_detection=True,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    cube = scene.add_entity(
        morph=gs.morphs.Box(size=CUBE_SIZE, pos=(args.cube_x, args.cube_y, args.cube_z)),
        material=gs.materials.Rigid(friction=args.cube_friction),
        surface=gs.surfaces.Default(color=(1.0, 0.3, 0.3, 1.0)),
    )
    so101 = scene.add_entity(gs.morphs.MJCF(file=str(xml), pos=(0.0, 0.0, 0.0)))
    cam_top = scene.add_camera(
        res=(640, 480), pos=(0.42, 0.34, 0.26), lookat=(0.15, 0.0, 0.08), fov=38, GUI=False,
    )
    cam_side = scene.add_camera(
        res=(640, 480),
        pos=(float(args.cube_x), float(args.cube_y - 0.32), float(args.cube_z + 0.09)),
        lookat=(float(args.cube_x), float(args.cube_y), float(args.cube_z + 0.03)),
        fov=50, GUI=False,
    )
    scene.build()

    dof_idx = np.arange(so101.n_dofs)
    so101.set_dofs_kp(KP[:so101.n_dofs], dof_idx)
    so101.set_dofs_kv(KV[:so101.n_dofs], dof_idx)
    home_deg = HOME_DEG[:so101.n_dofs]
    home_rad = np.deg2rad(home_deg)

    ee = so101.get_link("grasp_center")
    fixed_jaw = so101.get_link("gripper")
    moving_jaw = so101.get_link("moving_jaw_so101_v1")
    jaw_box_cfg = load_jaw_box_config(xml)
    cube_init = np.array([args.cube_x, args.cube_y, args.cube_z], dtype=np.float64)

    # --- wrist_roll joint limits (radians) from MJCF ---
    wrist_roll_range_rad = np.array([-2.7438, 2.8412])

    # ----- helpers -----

    def reset_scene():
        so101.set_qpos(home_rad)
        so101.control_dofs_position(home_rad, dof_idx)
        so101.zero_all_dofs_velocity()
        cube.set_pos(torch.tensor(cube_init, dtype=torch.float32, device=gs.device).unsqueeze(0))
        cube.set_quat(torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=gs.device).unsqueeze(0))
        cube.zero_all_dofs_velocity()
        for _ in range(args.settle_steps):
            scene.step()

    def solve_ik(pos, grip_deg, seed_rad, local_point=None):
        kwargs = dict(
            link=ee, pos=np.array(pos, dtype=np.float32),
            quat=None, init_qpos=seed_rad, max_solver_iters=50, damping=0.02,
        )
        if local_point is not None:
            kwargs["local_point"] = np.array(local_point, dtype=np.float32)
        q = to_numpy(so101.inverse_kinematics(**kwargs))
        q_deg = np.rad2deg(q)
        if so101.n_dofs >= 6:
            q_deg[5] = grip_deg
        return q_deg

    def get_jaw_box_world(link, cfg):
        lp, lq = to_numpy(link.get_pos()), to_numpy(link.get_quat())
        cw = transform_point(lp, lq, cfg["pos"])
        rot = quat_to_rotmat(lq)
        ta = np.zeros(3); ta[int(np.argmin(cfg["size"]))] = 1.0
        return {"center": cw, "axis": normalize(rot @ ta), "half_t": float(np.min(cfg["size"]))}

    def measure_jaw():
        fb = get_jaw_box_world(fixed_jaw, jaw_box_cfg["fixed_jaw_box"])
        mb = get_jaw_box_world(moving_jaw, jaw_box_cfg["moving_jaw_box"])
        f2m = mb["center"] - fb["center"]
        fi = fb["axis"] * np.sign(np.dot(f2m, fb["axis"]) or 1.0)
        mi = mb["axis"] * np.sign(np.dot(-f2m, mb["axis"]) or 1.0)
        f_inner = fb["center"] + fi * fb["half_t"]
        m_inner = mb["center"] + mi * mb["half_t"]
        mid = 0.5 * (f_inner + m_inner)
        gap_vec = m_inner - f_inner
        closing_axis = normalize(gap_vec) if np.linalg.norm(gap_vec) > 1e-6 else np.array([1., 0., 0.])
        return {
            "mid": mid,
            "delta_z": float(m_inner[2] - f_inner[2]),
            "jaw_gap": float(np.linalg.norm(gap_vec)),
            "closing_axis": closing_axis,
        }

    def compute_mid_local_point():
        jaw = measure_jaw()
        gc_pos = np.array(to_numpy(ee.get_pos()), dtype=np.float64)
        gc_quat = to_numpy(ee.get_quat())
        return quat_to_rotmat(gc_quat).T @ (jaw["mid"] - gc_pos)

    def settle_and_measure(q_deg, steps=3):
        q_rad = np.deg2rad(np.array(q_deg, dtype=np.float32))
        so101.set_qpos(q_rad)
        so101.control_dofs_position(q_rad, dof_idx)
        for _ in range(steps):
            scene.step()
        return measure_jaw()

    # ----- compute mid_local_point at reference pose -----
    reset_scene()
    ref_deg = home_deg.copy()
    if so101.n_dofs >= 6:
        ref_deg[5] = args.gripper_open
    ref_rad = np.deg2rad(np.array(ref_deg, dtype=np.float32))
    so101.set_qpos(ref_rad)
    so101.control_dofs_position(ref_rad, dof_idx)
    for _ in range(10):
        scene.step()
    mid_local = compute_mid_local_point()
    print(f"[init] mid_local_point = [{mid_local[0]:.6f}, {mid_local[1]:.6f}, {mid_local[2]:.6f}]")

    # ----- descent waypoints (position-only IK + seed chain + adaptive yaw snap) -----
    grasp_target_z = args.cube_z + args.grasp_offset_z + args.approach_z
    cube_top_z = args.cube_z + CUBE_SIZE[2] / 2.0
    pre_pos = cube_init + np.array([0.0, 0.0, PRE_HEIGHT])

    q_pre = solve_ik(pre_pos, args.gripper_open, home_rad, local_point=mid_local)
    seed = np.deg2rad(np.array(q_pre, dtype=np.float32))

    wrist_roll_lo = float(np.degrees(wrist_roll_range_rad[0]))
    wrist_roll_hi = float(np.degrees(wrist_roll_range_rad[1]))

    descent_z_step = abs(PRE_HEIGHT - grasp_target_z) / N_DESCENT
    snap_trigger = descent_z_step * SNAP_BUFFER_STEPS
    snap_started = False
    snap_start_wp = -1
    yaw_delta_total_deg = 0.0
    wrist_roll_before_snap = 0.0
    wrist_roll_target = 0.0
    yaw_snap_info = {}

    descent_wps = []
    descent_phases = []
    for i in range(N_DESCENT):
        frac = (i + 1) / N_DESCENT
        z = PRE_HEIGHT + (grasp_target_z - PRE_HEIGHT) * frac
        pos = cube_init + np.array([0.0, 0.0, z - args.cube_z])
        wp = solve_ik(pos, args.gripper_open, seed, local_point=mid_local)

        jaw_now = settle_and_measure(wp, steps=3)
        buffer_z = jaw_now["mid"][2] - cube_top_z

        if not snap_started and buffer_z < snap_trigger:
            snap_started = True
            snap_start_wp = i
            closing_xy = jaw_now["closing_axis"][:2]
            yaw_current = float(np.arctan2(closing_xy[1], closing_xy[0]))
            yaw_target = round(yaw_current / (np.pi / 2)) * (np.pi / 2)
            yaw_delta_total_deg = float(np.degrees(yaw_target - yaw_current))
            wrist_roll_before_snap = float(wp[WRIST_ROLL_IDX])
            wrist_roll_target = float(np.clip(
                wrist_roll_before_snap + yaw_delta_total_deg,
                wrist_roll_lo, wrist_roll_hi,
            ))
            remaining_wps = N_DESCENT - i
            print(f"[yaw_snap] triggered at wp {i}, buffer_z={buffer_z*1000:.1f}mm, "
                  f"yaw={np.degrees(yaw_current):.1f}° → {np.degrees(yaw_target):.1f}°, "
                  f"delta={yaw_delta_total_deg:.1f}°, {remaining_wps} steps to blend")

        if snap_started:
            remaining_wps = N_DESCENT - snap_start_wp
            progress = (i - snap_start_wp + 1) / remaining_wps
            blended_roll = wrist_roll_before_snap + progress * (wrist_roll_target - wrist_roll_before_snap)
            wp[WRIST_ROLL_IDX] = float(np.clip(blended_roll, wrist_roll_lo, wrist_roll_hi))
            descent_phases.append("approach_snap")
        else:
            descent_phases.append("approach")

        seed = np.deg2rad(np.array(wp, dtype=np.float32))
        descent_wps.append(wp)

    q_approach = descent_wps[-1]

    jaw_final_approach = settle_and_measure(q_approach, steps=5)
    closing_xy_final = jaw_final_approach["closing_axis"][:2]
    yaw_after = float(np.degrees(np.arctan2(closing_xy_final[1], closing_xy_final[0])))
    yaw_snap_info = {
        "triggered": snap_started,
        "trigger_wp": snap_start_wp,
        "snap_buffer_steps": SNAP_BUFFER_STEPS,
        "snap_trigger_m": snap_trigger,
        "yaw_delta_deg": yaw_delta_total_deg,
        "yaw_after_deg": yaw_after,
        "jaw_gap_after": jaw_final_approach["jaw_gap"],
        "delta_z_after": jaw_final_approach["delta_z"],
    }
    print(f"[yaw_snap] after blend: yaw={yaw_after:.1f}°, "
          f"gap={jaw_final_approach['jaw_gap']*1000:.1f}mm, "
          f"delta_z={jaw_final_approach['delta_z']*1000:.2f}mm")

    # ----- close + lift -----
    q_close = q_approach.copy()
    q_close[5] = args.gripper_close

    lift_wps = []
    seed = np.deg2rad(np.array(q_close, dtype=np.float32))
    for i in range(N_LIFT_WPS):
        frac = (i + 1) / N_LIFT_WPS
        z = grasp_target_z + (PRE_HEIGHT + args.cube_z - grasp_target_z) * frac
        pos = cube_init + np.array([0.0, 0.0, z - args.cube_z])
        wp = solve_ik(pos, args.gripper_close, seed, local_point=mid_local)
        seed = np.deg2rad(np.array(wp, dtype=np.float32))
        lift_wps.append(wp)

    # ----- assemble trajectory -----
    steps_per_wp = 3
    traj, phases = [], []

    traj += lerp(home_deg, q_pre, steps_per_wp)
    phases += ["move_pre"] * steps_per_wp

    prev = q_pre
    for wi, wp in enumerate(descent_wps):
        seg = lerp(prev, wp, steps_per_wp)
        traj += seg; phases += [descent_phases[wi]] * len(seg)
        prev = wp

    traj += [q_approach] * 2
    phases += ["approach_hold"] * 2

    traj += lerp(q_approach, q_close, N_CLOSE)
    phases += ["close"] * N_CLOSE

    traj += [q_close] * args.close_hold_steps
    phases += ["close_hold"] * args.close_hold_steps

    prev = q_close
    for wp in lift_wps:
        seg = lerp(prev, wp, steps_per_wp)
        traj += seg; phases += ["lift"] * len(seg)
        prev = wp

    print(f"[traj] total frames: {len(traj)}")

    # ----- execute -----
    reset_scene()
    out_root = Path(args.save) / args.exp_id
    out_root.mkdir(parents=True, exist_ok=True)
    png_dir = out_root / "pngs"
    png_dir.mkdir(parents=True, exist_ok=True)

    keep = {i for i, p in enumerate(phases)
            if p in {"approach", "approach_snap", "approach_hold", "close", "close_hold", "lift"}}
    frame_buffer = []

    for i, q_deg in enumerate(traj):
        so101.control_dofs_position(np.deg2rad(np.array(q_deg, dtype=np.float32)), dof_idx)
        scene.step()
        if i in keep:
            frame_buffer.append((i, phases[i],
                np.concatenate([render_camera(cam_top), render_camera(cam_side)], axis=1).astype(np.uint8)))

    for i, phase, img in frame_buffer:
        save_rgb_png(img, png_dir / f"f{i:03d}_{phase}.png")

    # ----- evaluate -----
    cube_pos_final = to_numpy(cube.get_pos())
    cube_quat_final = to_numpy(cube.get_quat())
    cube_shift = cube_pos_final - cube_init
    lift_z = float(cube_shift[2])
    tilt = box_tilt_deg(cube_quat_final)
    jaw_final = measure_jaw()

    success = lift_z > 0.02 and tilt < 30.0
    print(f"\n{'='*50}")
    print(f"  lift_z:     {lift_z*1000:.1f} mm  ({'OK' if lift_z > 0.02 else 'FAIL'})")
    print(f"  tilt:       {tilt:.1f} deg  ({'OK' if tilt < 30 else 'FAIL'})")
    print(f"  cube_shift: [{cube_shift[0]*1000:.1f}, {cube_shift[1]*1000:.1f}, {cube_shift[2]*1000:.1f}] mm")
    print(f"  jaw_gap:    {jaw_final['jaw_gap']*1000:.1f} mm")
    print(f"  delta_z:    {jaw_final['delta_z']*1000:.2f} mm")
    print(f"  SUCCESS:    {success}")
    print(f"{'='*50}")

    summary = {
        "exp_id": args.exp_id,
        "success": success,
        "cube_init": cube_init.tolist(),
        "cube_size": list(CUBE_SIZE),
        "cube_final_pos": cube_pos_final.tolist(),
        "cube_shift": cube_shift.tolist(),
        "lift_z_m": lift_z,
        "tilt_deg": tilt,
        "yaw_snap": yaw_snap_info,
        "jaw_final": {
            "jaw_gap": jaw_final["jaw_gap"],
            "delta_z": jaw_final["delta_z"],
        },
        "params": {
            "cube_x": args.cube_x, "cube_y": args.cube_y, "cube_z": args.cube_z,
            "gripper_open": args.gripper_open, "gripper_close": args.gripper_close,
            "grasp_offset_z": args.grasp_offset_z, "approach_z": args.approach_z,
            "grasp_target_z": grasp_target_z,
        },
        "mid_local_point": mid_local.tolist(),
        "traj_frames": len(traj),
        "xml_path": str(xml),
    }

    out_path = out_root / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[saved] {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
