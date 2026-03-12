"""
SO-101 Workspace Feasibility Mapper (v2)

Pure arm reachability scan — no cube in scene, no orientation constraints.

For every point on a 3D grid, solve position-only IK and measure:
  - pos_error:  jaw midpoint vs target  (reachability)
  - delta_z:    height diff between two jaw inner surfaces  (levelness)
  - jaw_gap:    distance between inner surfaces  (can it fit a cube?)

Then filter by a single --cube-size to output the feasible grasp zone
and recommended cube placement center.

Strategy:
  For each (x, y) column, descend from z_max to z_min using the previous
  z-level's IK solution as seed (mimics a real approach trajectory).
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
# Utility helpers (shared with 33/34 scripts)
# ---------------------------------------------------------------------------


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


def parse_range(s: str) -> np.ndarray:
    """Parse 'start:stop:step' or 'v1,v2,...' into a sorted array."""
    if ":" in s:
        parts = s.split(":")
        start, stop, step = float(parts[0]), float(parts[1]), float(parts[2])
        return np.arange(start, stop + step * 0.49, step)
    return np.array(sorted(float(v) for v in s.split(",")))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HOME_DEG = np.array([0.0, -30.0, 90.0, -60.0, 0.0, 0.0], dtype=np.float32)
KP = np.array([500.0, 500.0, 400.0, 400.0, 300.0, 200.0], dtype=np.float32)
KV = np.array([50.0, 50.0, 40.0, 40.0, 30.0, 20.0], dtype=np.float32)

POS_ERROR_THRESH = 0.005  # 5 mm
DELTA_Z_THRESH = 0.004  # 4 mm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ensure_display()

    ap = argparse.ArgumentParser(
        description="SO-101 workspace feasibility mapper v2 (pure arm reachability)"
    )
    ap.add_argument("--exp-id", default="workspace_map")
    ap.add_argument("--xml", default=None)
    ap.add_argument("--save", default="/output")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--sim-dt", type=float, default=1.0 / 30.0)
    ap.add_argument("--sim-substeps", type=int, default=4)
    ap.add_argument("--settle-steps", type=int, default=8)
    ap.add_argument("--gripper-open", type=float, default=25.0)
    ap.add_argument(
        "--grid-x",
        default="0.08:0.26:0.01",
        help="X range as start:stop:step or v1,v2,...",
    )
    ap.add_argument(
        "--grid-y",
        default="-0.10:0.10:0.01",
        help="Y range as start:stop:step or v1,v2,...",
    )
    ap.add_argument(
        "--cube-size",
        type=float,
        default=0.03,
        help="Cube full side length (m). Grid Z is fixed to cube_size/2 (resting on ground).",
    )
    ap.add_argument(
        "--pos-error-thresh",
        type=float,
        default=POS_ERROR_THRESH,
        help="Position error threshold (m) for 'reachable'",
    )
    ap.add_argument(
        "--delta-z-thresh",
        type=float,
        default=DELTA_Z_THRESH,
        help="Jaw delta-z threshold (m) for 'jaws_level'",
    )
    args = ap.parse_args()

    xml = find_so101_xml(args.xml)
    if xml is None:
        raise RuntimeError("SO101 xml not found")

    xs = parse_range(args.grid_x)
    ys = parse_range(args.grid_y)
    grasp_z = args.cube_size / 2.0

    n_total = len(xs) * len(ys)
    n_columns = n_total
    print(f"[config] grid: X={len(xs)}  Y={len(ys)}  -> {n_total} pts")
    print(f"[config] X: [{xs[0]:.3f} .. {xs[-1]:.3f}]")
    print(f"[config] Y: [{ys[0]:.3f} .. {ys[-1]:.3f}]")
    print(f"[config] Z (grasp height): {grasp_z:.4f}  (cube_size/2)")
    print(f"[config] cube_size: {args.cube_size}m")
    print(
        f"[config] thresholds: pos_error={args.pos_error_thresh}m  delta_z={args.delta_z_thresh}m"
    )

    # -----------------------------------------------------------------------
    # Genesis setup
    # -----------------------------------------------------------------------
    import genesis as gs

    gs.init(backend=(gs.cpu if args.cpu else gs.gpu), logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=args.sim_dt, substeps=args.sim_substeps),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            enable_joint_limit=True,
            box_box_detection=True,
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

    # -----------------------------------------------------------------------
    # Jaw geometry helpers
    # -----------------------------------------------------------------------

    def reset_home():
        so101.set_qpos(home_rad)
        so101.control_dofs_position(home_rad, dof_idx)
        so101.zero_all_dofs_velocity()
        for _ in range(args.settle_steps):
            scene.step()

    def solve_ik(pos, grip_deg, seed_rad, local_point=None):
        kwargs = dict(
            link=ee,
            pos=np.array(pos, dtype=np.float32),
            quat=None,
            init_qpos=seed_rad,
            max_solver_iters=50,
            damping=0.02,
        )
        if local_point is not None:
            kwargs["local_point"] = np.array(local_point, dtype=np.float32)
        # No rot_mask, no quat — pure position IK only.
        q = to_numpy(so101.inverse_kinematics(**kwargs))
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

    def measure_jaw_state():
        fixed_box = get_jaw_box_world(fixed_jaw, jaw_box_cfg["fixed_jaw_box"])
        moving_box = get_jaw_box_world(moving_jaw, jaw_box_cfg["moving_jaw_box"])
        fa = fixed_box["thickness_axis_world"]
        ma = moving_box["thickness_axis_world"]
        f2m = moving_box["center_world"] - fixed_box["center_world"]
        m2f = -f2m
        fi = fa * np.sign(np.dot(f2m, fa) or 1.0)
        mi = ma * np.sign(np.dot(m2f, ma) or 1.0)
        fixed_inner = fixed_box["center_world"] + fi * fixed_box["half_thickness"]
        moving_inner = moving_box["center_world"] + mi * moving_box["half_thickness"]
        mid = 0.5 * (fixed_inner + moving_inner)

        gap_vec = moving_inner - fixed_inner
        jaw_gap = float(np.linalg.norm(gap_vec))
        closing_axis = (
            normalize(gap_vec) if jaw_gap > 1e-6 else np.array([1.0, 0.0, 0.0])
        )
        closing_axis_tilt_deg = float(
            np.degrees(np.arcsin(np.clip(abs(closing_axis[2]), 0, 1)))
        )
        delta_z = float(moving_inner[2] - fixed_inner[2])

        return {
            "mid": mid,
            "fixed_inner": fixed_inner,
            "moving_inner": moving_inner,
            "delta_z": delta_z,
            "jaw_gap": jaw_gap,
            "closing_axis": closing_axis,
            "closing_axis_tilt_deg": closing_axis_tilt_deg,
        }

    def compute_mid_local_point():
        jaw = measure_jaw_state()
        mid_world = np.array(jaw["mid"], dtype=np.float64)
        gc_pos = np.array(to_numpy(ee.get_pos()), dtype=np.float64)
        gc_quat = to_numpy(ee.get_quat())
        rot = quat_to_rotmat(gc_quat)
        return rot.T @ (mid_world - gc_pos)

    # -----------------------------------------------------------------------
    # Compute mid_local_point at reference pose (home + gripper_open)
    # -----------------------------------------------------------------------
    reset_home()
    q_ref = home_deg.copy()
    if so101.n_dofs >= 6:
        q_ref[5] = args.gripper_open
    q_ref_rad = np.deg2rad(np.array(q_ref, dtype=np.float32))
    so101.set_qpos(q_ref_rad)
    so101.control_dofs_position(q_ref_rad, dof_idx)
    for _ in range(10):
        scene.step()
    mid_local_pt = compute_mid_local_point()
    print(
        f"[init] mid_local_point = [{mid_local_pt[0]:.6f}, {mid_local_pt[1]:.6f}, {mid_local_pt[2]:.6f}]"
    )

    # -----------------------------------------------------------------------
    # Grid scan
    # -----------------------------------------------------------------------
    results = []
    col_idx = 0
    t_start = time.time()

    for xi, x in enumerate(xs):
        for yi, y in enumerate(ys):
            col_idx += 1
            reset_home()

            target = np.array([x, y, grasp_z], dtype=np.float64)
            q_deg = solve_ik(
                target, args.gripper_open, home_rad, local_point=mid_local_pt
            )
            q_rad = np.deg2rad(np.array(q_deg, dtype=np.float32))

            so101.set_qpos(q_rad)
            so101.control_dofs_position(q_rad, dof_idx)
            for _ in range(args.settle_steps):
                scene.step()

            jaw = measure_jaw_state()
            mid_actual = jaw["mid"]
            gc_pos = np.array(to_numpy(ee.get_pos()), dtype=np.float64)

            err_3d = float(np.linalg.norm(mid_actual - target))
            err_xy = float(np.linalg.norm(mid_actual[:2] - target[:2]))
            err_z = float(abs(mid_actual[2] - target[2]))

            reachable = err_3d < args.pos_error_thresh
            jaws_level = abs(jaw["delta_z"]) < args.delta_z_thresh

            results.append(
                {
                    "x": round(float(x), 4),
                    "y": round(float(y), 4),
                    "z": round(float(grasp_z), 4),
                    "mid_actual": [round(float(v), 5) for v in mid_actual.tolist()],
                    "gc_pos": [round(float(v), 5) for v in gc_pos.tolist()],
                    "pos_error_3d": round(err_3d, 5),
                    "pos_error_xy": round(err_xy, 5),
                    "pos_error_z": round(err_z, 5),
                    "delta_z": round(float(jaw["delta_z"]), 5),
                    "jaw_gap": round(float(jaw["jaw_gap"]), 5),
                    "closing_tilt_deg": round(jaw["closing_axis_tilt_deg"], 2),
                    "reachable": reachable,
                    "jaws_level": jaws_level,
                }
            )

            elapsed = time.time() - t_start
            eta = elapsed / col_idx * (n_columns - col_idx)
            print(
                f"  col {col_idx:4d}/{n_columns}  ({x:+.3f}, {y:+.3f})  "
                f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s ETA]"
            )

    scan_time = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Grid scan complete: {len(results)} points in {scan_time:.1f}s")

    # -----------------------------------------------------------------------
    # Analysis — single cube_size filter
    # -----------------------------------------------------------------------

    cube_size = args.cube_size
    n_reachable = sum(1 for r in results if r["reachable"])
    n_level = sum(1 for r in results if r["jaws_level"])
    n_reach_and_level = sum(1 for r in results if r["reachable"] and r["jaws_level"])
    all_gaps = [r["jaw_gap"] for r in results]

    feasible = [
        r
        for r in results
        if r["reachable"] and r["jaws_level"] and r["jaw_gap"] > cube_size
    ]

    print(f"\n  Total scanned:    {len(results)}")
    print(
        f"  Reachable:        {n_reachable}  ({n_reachable / len(results) * 100:.1f}%)"
    )
    print(f"  Jaws level:       {n_level}  ({n_level / len(results) * 100:.1f}%)")
    print(f"  Reach + level:    {n_reach_and_level}")
    print(f"  Grasp-feasible (gap > {cube_size}m): {len(feasible)}")

    # Feasible zone bounds + recommended cube center
    feasible_zone = None
    recommended_cube_center = None
    if feasible:
        f_xs = [r["x"] for r in feasible]
        f_ys = [r["y"] for r in feasible]
        feasible_zone = {
            "x_min": round(min(f_xs), 4),
            "x_max": round(max(f_xs), 4),
            "y_min": round(min(f_ys), 4),
            "y_max": round(max(f_ys), 4),
        }
        recommended_cube_center = [
            round(float(np.mean(f_xs)), 4),
            round(float(np.mean(f_ys)), 4),
            round(grasp_z, 4),
        ]
        print(f"\n  Feasible zone (XY at z={grasp_z:.4f}):")
        print(f"    X: [{feasible_zone['x_min']:.3f}, {feasible_zone['x_max']:.3f}]")
        print(f"    Y: [{feasible_zone['y_min']:.3f}, {feasible_zone['y_max']:.3f}]")
        print(f"\n  Recommended cube center: {recommended_cube_center}")
    else:
        print(
            f"\n  *** NO feasible points for cube_size={cube_size}m at z={grasp_z:.4f} ***"
        )

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    summary = {
        "exp_id": args.exp_id,
        "xml_path": str(xml),
        "scan_time_s": round(scan_time, 1),
        "grid": {
            "x": [round(float(v), 4) for v in xs.tolist()],
            "y": [round(float(v), 4) for v in ys.tolist()],
            "grasp_z": round(grasp_z, 4),
            "total_points": len(results),
        },
        "cube_size": cube_size,
        "thresholds": {
            "pos_error_m": args.pos_error_thresh,
            "delta_z_m": args.delta_z_thresh,
        },
        "gripper_open_deg": float(args.gripper_open),
        "mid_local_point": [round(float(v), 6) for v in mid_local_pt.tolist()],
        "overall_stats": {
            "reachable_count": n_reachable,
            "reachable_pct": round(n_reachable / len(results) * 100, 1),
            "jaws_level_count": n_level,
            "jaws_level_pct": round(n_level / len(results) * 100, 1),
            "reach_and_level_count": n_reach_and_level,
            "reach_and_level_pct": round(n_reach_and_level / len(results) * 100, 1),
            "feasible_count": len(feasible),
            "feasible_pct": round(len(feasible) / len(results) * 100, 1),
            "jaw_gap_min": round(min(all_gaps), 5),
            "jaw_gap_max": round(max(all_gaps), 5),
            "jaw_gap_mean": round(float(np.mean(all_gaps)), 5),
        },
        "feasible_zone": feasible_zone,
        "recommended_cube_center": recommended_cube_center,
        "grid_points": results,
    }

    out_root = Path(args.save) / args.exp_id
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / "workspace_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[saved] {out_path}")

    compact = {k: v for k, v in summary.items() if k != "grid_points"}
    print("\n" + json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
