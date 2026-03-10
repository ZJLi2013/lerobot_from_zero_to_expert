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
    tcp_local_point = None
    tcp_half_span_world = None
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
        return tcp_pos, delta

    stage("2/4 Run no-contact grid")
    print(f"points={len(points)} repeats={args.repeats} target_z={args.target_z:.4f}")
    rows = []
    for x, y in points:
        target = np.array([x, y, args.target_z], dtype=np.float32)
        samples = []
        for _ in range(args.repeats):
            reset_home()
            tcp_pos, delta = solve_and_measure(target)
            samples.append(
                {
                    "ik_target": [float(v) for v in target.tolist()],
                    "tcp_actual": [float(v) for v in tcp_pos.tolist()],
                    "tcp_offset": [float(v) for v in delta.tolist()],
                }
            )
        deltas = np.array([s["tcp_offset"] for s in samples], dtype=np.float64)
        mean_delta = deltas.mean(axis=0)
        std_delta = deltas.std(axis=0)
        row = {
            "target_x": float(x),
            "target_y": float(y),
            "target_z": float(args.target_z),
            "repeats": args.repeats,
            "tcp_offset_mean": [float(v) for v in mean_delta.tolist()],
            "tcp_offset_std": [float(v) for v in std_delta.tolist()],
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
    summary = {
        "exp_id": args.exp_id,
        "xml_path": str(xml_path),
        "ee_link": ee_name,
        "ik_supports_local_point": bool(ik_supports_local_point),
        "target_z": float(args.target_z),
        "grid_x": xs,
        "grid_y": ys,
        "num_points": len(points),
        "repeats": int(args.repeats),
        "tcp_offset_global_mean": [float(v) for v in mean_all.mean(axis=0).tolist()],
        "tcp_offset_global_std_over_points": [float(v) for v in mean_all.std(axis=0).tolist()],
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

