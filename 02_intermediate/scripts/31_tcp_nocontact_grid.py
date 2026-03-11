"""
No-contact TCP offset grid test for SO-101 MJCF.

Purpose:
- Remove cube/contact effects and measure tcp_actual - ik_target directly.
- Reuse EE/TCP handling from 3_grasp_experiment.py.
"""

import argparse
import inspect
import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET
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


def parse_csv_floats(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_vec(s):
    return np.array([float(v) for v in s.split()], dtype=np.float64)


def normalize(v):
    v = np.array(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


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


def transform_point(link_pos, link_quat, local_pos):
    return np.array(link_pos, dtype=np.float64) + quat_to_rotmat(link_quat) @ np.array(
        local_pos, dtype=np.float64
    )


def find_so101_xml(user_path=None):
    if user_path:
        p = Path(user_path)
        if p.exists():
            return p
        return None
    candidates = [
        Path("assets/so101_new_calib_v4.xml"),
        Path("assets/so101_new_calib.xml"),
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


def stage(name):
    print(f"\n{'-' * 60}\n[{name}]\n{'-' * 60}")


HOME_DEG = np.array([0.0, -30.0, 90.0, -60.0, 0.0, 0.0], dtype=np.float32)
KP = np.array([500.0, 500.0, 400.0, 400.0, 300.0, 200.0])
KV = np.array([50.0, 50.0, 40.0, 40.0, 30.0, 20.0])


def main():
    ensure_display()

    parser = argparse.ArgumentParser(description="SO-101 no-contact TCP grid test")
    parser.add_argument("--exp-id", default="tcp_nocontact_grid")
    parser.add_argument("--xml", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save", default="/output")
    parser.add_argument("--target-z", type=float, default=0.06)
    parser.add_argument("--grid-x", default="0.156,0.160,0.164")
    parser.add_argument("--grid-y", default="-0.004,0.000,0.004")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--sim-dt", type=float, default=1.0 / 30.0)
    parser.add_argument("--sim-substeps", type=int, default=4)
    args = parser.parse_args()

    xs = parse_csv_floats(args.grid_x)
    ys = parse_csv_floats(args.grid_y)
    points = [(x, y) for x in xs for y in ys]
    if len(points) > 10:
        raise ValueError(f"Grid has {len(points)} points (>10). Reduce grid-x/grid-y.")

    stage("1/4 Load robot")
    xml_path = find_so101_xml(args.xml)
    if xml_path is None:
        raise RuntimeError("SO-101 xml not found. Pass --xml.")
    print(f"xml: {xml_path}")
    jaw_box_cfg = load_jaw_box_config(xml_path)

    import genesis as gs
    import genesis.utils.geom as gu

    backend = gs.cpu if args.cpu else gs.gpu
    gs.init(backend=backend, logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=args.sim_dt, substeps=args.sim_substeps),
        rigid_options=gs.options.RigidOptions(enable_collision=True, enable_joint_limit=True, box_box_detection=True),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    so101 = scene.add_entity(gs.morphs.MJCF(file=str(xml_path), pos=(0.0, 0.0, 0.0)))
    scene.build()

    n_dofs = so101.n_dofs
    dof_idx = np.arange(n_dofs)
    so101.set_dofs_kp(KP[:n_dofs], dof_idx)
    so101.set_dofs_kv(KV[:n_dofs], dof_idx)
    home_deg = HOME_DEG[:n_dofs]
    home_rad = np.deg2rad(home_deg)

    ee_link = None
    ee_name = None
    for candidate in [
        "grasp_center",
        "moving_jaw_so101_v1",
        "Moving_Jaw",
        "Fixed_Jaw",
        "gripper_link",
        "gripperframe",
        "gripper",
    ]:
        try:
            ee_link = so101.get_link(candidate)
            ee_name = candidate
            print(f"ee_link: {candidate}")
            break
        except Exception:
            pass
    if ee_link is None:
        raise RuntimeError("EE link not found")

    ik_supports_local_point = "local_point" in inspect.signature(so101.inverse_kinematics).parameters
    gripper_ref = None
    fixed_jaw_link = None
    moving_jaw_link = None
    tcp_local_point = None
    tcp_half_span_world = None
    try:
        fixed_jaw_link = so101.get_link("gripper")
    except Exception:
        fixed_jaw_link = None
    try:
        moving_jaw_link = so101.get_link("moving_jaw_so101_v1")
    except Exception:
        moving_jaw_link = None
    if ee_name == "moving_jaw_so101_v1":
        try:
            gripper_ref = so101.get_link("gripper")
            jaw_pos = to_numpy(ee_link.get_pos())
            jaw_quat = to_numpy(ee_link.get_quat())
            grip_pos = to_numpy(gripper_ref.get_pos())
            tcp_world = 0.5 * (jaw_pos + grip_pos)
            tcp_half_span_world = 0.5 * (grip_pos - jaw_pos)
            if ik_supports_local_point:
                tcp_local_point = gu.inv_transform_by_quat(tcp_world - jaw_pos, jaw_quat)
        except Exception as e:
            print(f"tcp proxy unavailable: {e}")
    elif ee_name == "grasp_center":
        print("using grasp_center as fixed tcp")

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

    def reset_home():
        so101.set_qpos(home_rad)
        so101.control_dofs_position(home_rad, dof_idx)
        so101.zero_all_dofs_velocity()
        for _ in range(args.settle_steps):
            scene.step()

    def solve_and_measure(target_xyz):
        if ik_supports_local_point and tcp_local_point is not None:
            q = so101.inverse_kinematics(
                link=ee_link,
                pos=target_xyz,
                quat=None,
                local_point=tcp_local_point,
                init_qpos=home_rad,
                max_solver_iters=50,
                damping=0.02,
            )
        elif gripper_ref is not None and tcp_half_span_world is not None:
            q = so101.inverse_kinematics_multilink(
                links=[ee_link, gripper_ref],
                poss=[target_xyz - tcp_half_span_world, target_xyz + tcp_half_span_world],
                quats=None,
                init_qpos=home_rad,
                max_solver_iters=50,
                damping=0.02,
            )
        else:
            q = so101.inverse_kinematics(
                link=ee_link,
                pos=target_xyz,
                quat=None,
                init_qpos=home_rad,
                max_solver_iters=50,
                damping=0.02,
            )
        q_np = to_numpy(q) if hasattr(q, "cpu") else np.array(q)
        so101.set_qpos(q_np)
        so101.control_dofs_position(q_np, dof_idx)
        for _ in range(args.settle_steps):
            scene.step()

        ee_pos = to_numpy(ee_link.get_pos())
        ee_quat = to_numpy(ee_link.get_quat())
        if ik_supports_local_point and tcp_local_point is not None:
            tcp_pos = ee_pos + gu.transform_by_quat(tcp_local_point, ee_quat)
        elif gripper_ref is not None:
            grip_pos = to_numpy(gripper_ref.get_pos())
            tcp_pos = 0.5 * (ee_pos + grip_pos)
        else:
            tcp_pos = ee_pos
        delta = tcp_pos - target_xyz
        delta_jaw = None
        delta_jaw_link_origin = None
        delta_jaw_inner_surface = None
        if fixed_jaw_link is not None and moving_jaw_link is not None:
            z_fixed = float(to_numpy(fixed_jaw_link.get_pos())[2])
            z_moving = float(to_numpy(moving_jaw_link.get_pos())[2])
            delta_jaw_link_origin = float(z_moving - z_fixed)

            fixed_box = get_jaw_box_world(fixed_jaw_link, jaw_box_cfg["fixed_jaw_box"])
            moving_box = get_jaw_box_world(moving_jaw_link, jaw_box_cfg["moving_jaw_box"])
            fixed_axis = fixed_box["thickness_axis_world"]
            moving_axis = moving_box["thickness_axis_world"]
            fixed_to_moving = moving_box["center_world"] - fixed_box["center_world"]
            moving_to_fixed = fixed_box["center_world"] - moving_box["center_world"]
            fixed_inward = fixed_axis * np.sign(np.dot(fixed_to_moving, fixed_axis) or 1.0)
            moving_inward = moving_axis * np.sign(np.dot(moving_to_fixed, moving_axis) or 1.0)
            fixed_inner = fixed_box["center_world"] + fixed_inward * fixed_box["half_thickness"]
            moving_inner = moving_box["center_world"] + moving_inward * moving_box["half_thickness"]
            delta_jaw_inner_surface = float(moving_inner[2] - fixed_inner[2])
            # Backward-compatible alias now points to primary inner-surface metric.
            delta_jaw = float(delta_jaw_inner_surface)
        return tcp_pos, delta, delta_jaw, delta_jaw_link_origin, delta_jaw_inner_surface

    stage("2/4 Run no-contact grid")
    print(f"points={len(points)} repeats={args.repeats} target_z={args.target_z:.4f}")
    rows = []
    for x, y in points:
        target = np.array([x, y, args.target_z], dtype=np.float32)
        samples = []
        for _ in range(args.repeats):
            reset_home()
            tcp_pos, delta, delta_jaw, delta_jaw_link_origin, delta_jaw_inner_surface = solve_and_measure(target)
            samples.append(
                {
                    "ik_target": [float(v) for v in target.tolist()],
                    "tcp_actual": [float(v) for v in tcp_pos.tolist()],
                    "tcp_offset": [float(v) for v in delta.tolist()],
                    "delta_jaw": delta_jaw,
                    "delta_jaw_link_origin": delta_jaw_link_origin,
                    "delta_jaw_inner_surface": delta_jaw_inner_surface,
                }
            )
        deltas = np.array([s["tcp_offset"] for s in samples], dtype=np.float64)
        mean_delta = deltas.mean(axis=0)
        std_delta = deltas.std(axis=0)
        jaw_vals = np.array(
            [s["delta_jaw"] for s in samples if s["delta_jaw"] is not None], dtype=np.float64
        )
        jaw_vals_link = np.array(
            [s["delta_jaw_link_origin"] for s in samples if s["delta_jaw_link_origin"] is not None],
            dtype=np.float64,
        )
        jaw_vals_surface = np.array(
            [s["delta_jaw_inner_surface"] for s in samples if s["delta_jaw_inner_surface"] is not None],
            dtype=np.float64,
        )
        row = {
            "target_x": float(x),
            "target_y": float(y),
            "target_z": float(args.target_z),
            "repeats": args.repeats,
            "tcp_offset_mean": [float(v) for v in mean_delta.tolist()],
            "tcp_offset_std": [float(v) for v in std_delta.tolist()],
            "delta_jaw_mean": float(jaw_vals.mean()) if jaw_vals.size > 0 else None,
            "delta_jaw_std": float(jaw_vals.std()) if jaw_vals.size > 0 else None,
            "delta_jaw_link_origin_mean": float(jaw_vals_link.mean()) if jaw_vals_link.size > 0 else None,
            "delta_jaw_link_origin_std": float(jaw_vals_link.std()) if jaw_vals_link.size > 0 else None,
            "delta_jaw_inner_surface_mean": float(jaw_vals_surface.mean()) if jaw_vals_surface.size > 0 else None,
            "delta_jaw_inner_surface_std": float(jaw_vals_surface.std()) if jaw_vals_surface.size > 0 else None,
            "samples": samples,
        }
        rows.append(row)
        print(
            f"target=({x:+.3f},{y:+.3f},{args.target_z:+.3f}) "
            f"mean=({mean_delta[0]:+.4f},{mean_delta[1]:+.4f},{mean_delta[2]:+.4f}) "
            f"std=({std_delta[0]:.4f},{std_delta[1]:.4f},{std_delta[2]:.4f})"
        )

    stage("3/4 Save")
    out_dir = Path(args.save) / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_all = np.array([r["tcp_offset_mean"] for r in rows], dtype=np.float64)
    jaw_all = np.array(
        [r["delta_jaw_mean"] for r in rows if r.get("delta_jaw_mean") is not None],
        dtype=np.float64,
    )
    jaw_all_link = np.array(
        [r["delta_jaw_link_origin_mean"] for r in rows if r.get("delta_jaw_link_origin_mean") is not None],
        dtype=np.float64,
    )
    jaw_all_surface = np.array(
        [r["delta_jaw_inner_surface_mean"] for r in rows if r.get("delta_jaw_inner_surface_mean") is not None],
        dtype=np.float64,
    )
    summary = {
        "exp_id": args.exp_id,
        "xml_path": str(xml_path),
        "ee_link": ee_name,
        "jaw_links": {
            "fixed": "gripper" if fixed_jaw_link is not None else None,
            "moving": "moving_jaw_so101_v1" if moving_jaw_link is not None else None,
        },
        "ik_supports_local_point": bool(ik_supports_local_point),
        "jaw_reference_primary": "inner_surface_from_fixed_and_moving_jaw_box",
        "target_z": float(args.target_z),
        "grid_x": xs,
        "grid_y": ys,
        "num_points": len(points),
        "repeats": int(args.repeats),
        "tcp_offset_global_mean": [float(v) for v in mean_all.mean(axis=0).tolist()],
        "tcp_offset_global_std_over_points": [float(v) for v in mean_all.std(axis=0).tolist()],
        "delta_jaw_global_mean": float(jaw_all.mean()) if jaw_all.size > 0 else None,
        "delta_jaw_global_std_over_points": float(jaw_all.std()) if jaw_all.size > 0 else None,
        "delta_jaw_link_origin_global_mean": float(jaw_all_link.mean()) if jaw_all_link.size > 0 else None,
        "delta_jaw_link_origin_global_std_over_points": float(jaw_all_link.std()) if jaw_all_link.size > 0 else None,
        "delta_jaw_inner_surface_global_mean": float(jaw_all_surface.mean()) if jaw_all_surface.size > 0 else None,
        "delta_jaw_inner_surface_global_std_over_points": float(jaw_all_surface.std()) if jaw_all_surface.size > 0 else None,
        "point_results": rows,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"saved: {out_dir / 'summary.json'}")

    stage("4/4 Done")
    print(f"global_mean={summary['tcp_offset_global_mean']}")
    print(f"global_std_points={summary['tcp_offset_global_std_over_points']}")


if __name__ == "__main__":
    main()

