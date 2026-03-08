"""
SO-101 offset auto-tune with multiple metrics.

Metrics:
  - delta_z: close_hold z - close_start z (legacy, for backward compat)
  - centering_error: 3D distance from jaw midpoint to cube center at approach end
  - approach_contact: cube displacement from init at approach end (should be ~0)

Usage:
  python 10_auto_tune.py --xml assets/so101_new_calib_v3_jawbox.xml \
      --cube-friction 1.5 --save /output --exp-id T2
"""

import argparse
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
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
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
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


def save_png(arr, path):
    try:
        from PIL import Image
        Image.fromarray(arr).save(path)
    except ImportError:
        import imageio.v2 as imageio
        imageio.imwrite(path, arr)


def parse_csv(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_vec(text):
    return np.array([float(x) for x in text.split()], dtype=np.float64)


def quat_to_rotmat(quat_wxyz):
    w, x, y, z = quat_wxyz
    n = np.linalg.norm(quat_wxyz)
    if n == 0:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = quat_wxyz / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def transform_point(link_pos, link_quat, local_pos):
    rot = quat_to_rotmat(link_quat)
    return link_pos + rot @ local_pos


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


def find_xml(user_path):
    if user_path:
        p = Path(user_path)
        if p.exists():
            return p
    for c in [Path("assets/so101_new_calib.xml"),
              Path("02_intermediate/scripts/assets/so101_new_calib.xml")]:
        if c.exists():
            return c
    try:
        from huggingface_hub import snapshot_download
        d = snapshot_download(repo_type="dataset", repo_id="Genesis-Intelligence/assets",
                              allow_patterns="SO101/*", max_workers=1)
        p = Path(d) / "SO101" / "so101_new_calib.xml"
        if p.exists():
            return p
    except Exception:
        pass
    return None


JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]
HOME_DEG = np.array([0.0, -30.0, 90.0, -60.0, 0.0, 0.0], dtype=np.float32)
IK_QUAT_DOWN = np.array([0.0, 1.0, 0.0, 0.0])


def main():
    ensure_display()

    p = argparse.ArgumentParser(description="SO-101 offset auto-tune")
    p.add_argument("--xml", default=None)
    p.add_argument("--exp-id", default="T1")
    p.add_argument("--save", default="/output")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--gripper-open", type=float, default=30.0)
    p.add_argument("--gripper-close", type=float, default=-10.0)
    p.add_argument("--approach-z", type=float, default=0.012)
    p.add_argument("--cube-x", type=float, default=0.16)
    p.add_argument("--cube-y", type=float, default=0.0)
    p.add_argument("--cube-z", type=float, default=0.015)
    p.add_argument("--cube-friction", type=float, default=None)
    p.add_argument("--x-candidates", default="-0.008,-0.004,0.0,0.004,0.008")
    p.add_argument("--y-candidates", default="-0.008,-0.004,0.0,0.004,0.008")
    p.add_argument("--z-candidates", default="-0.010")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--trial-steps", type=int, default=90)
    p.add_argument("--contact-threshold", type=float, default=0.003,
                   help="Max cube displacement (m) to consider approach 'clean'")
    args = p.parse_args()

    xml_path = find_xml(args.xml)
    if not xml_path:
        print("XML not found"); sys.exit(1)
    print(f"XML: {xml_path}")
    jaw_box_cfg = load_jaw_box_config(xml_path)

    import torch
    import genesis as gs

    backend = gs.cpu if args.cpu else gs.gpu
    gs.init(backend=backend, logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / args.fps, substeps=4),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True, enable_joint_limit=True, box_box_detection=True,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())

    cube_kw = dict(
        morph=gs.morphs.Box(size=(0.03, 0.03, 0.03),
                            pos=(args.cube_x, args.cube_y, args.cube_z)),
        surface=gs.surfaces.Default(color=(1.0, 0.3, 0.3, 1.0)),
    )
    if args.cube_friction is not None:
        cube_kw["material"] = gs.materials.Rigid(friction=args.cube_friction)
        print(f"  cube friction={args.cube_friction}")
    cube = scene.add_entity(**cube_kw)
    so101 = scene.add_entity(gs.morphs.MJCF(file=str(xml_path), pos=(0, 0, 0)))

    cam_up = scene.add_camera(res=(640, 480), pos=(0.42, 0.34, 0.26),
                              lookat=(0.15, 0.0, 0.08), fov=38, GUI=False)
    cam_side = scene.add_camera(res=(640, 480), pos=(0.5, -0.4, 0.3),
                                lookat=(0.15, 0.0, 0.1), fov=45, GUI=False)
    scene.build()

    home_rad = np.deg2rad(HOME_DEG)
    dof_idx = [so101.get_joint(n).dof_idx_local for n in JOINT_NAMES]
    n_dofs = len(dof_idx)

    so101.set_dofs_kp(np.array([500, 500, 400, 400, 300, 200], dtype=np.float32), dof_idx)
    so101.set_dofs_kv(np.array([50, 50, 40, 40, 30, 20], dtype=np.float32), dof_idx)
    so101.set_qpos(home_rad)
    so101.control_dofs_position(home_rad, dof_idx)
    for _ in range(60):
        scene.step()

    ee_link = so101.get_link("grasp_center")  # IK target = grasp_center
    fixed_jaw_link = so101.get_link(jaw_box_cfg["fixed_jaw_box"]["body_name"])
    moving_jaw_link = so101.get_link(jaw_box_cfg["moving_jaw_box"]["body_name"])
    cube_half_extent = 0.03 / 2.0

    def get_jaw_box_world(link, cfg):
        link_pos = to_numpy(link.get_pos())
        link_quat = to_numpy(link.get_quat())
        center_world = transform_point(link_pos, link_quat, cfg["pos"])
        rot = quat_to_rotmat(link_quat)
        thickness_axis_local = np.zeros(3, dtype=np.float64)
        thickness_axis_local[int(np.argmin(cfg["size"]))] = 1.0
        thickness_axis_world = rot @ thickness_axis_local
        thickness_axis_world = thickness_axis_world / np.linalg.norm(thickness_axis_world)
        return {
            "center_world": center_world,
            "thickness_axis_world": thickness_axis_world,
            "half_thickness": float(np.min(cfg["size"])),
        }

    def compute_jaw_metrics(cube_center_world):
        fixed_box = get_jaw_box_world(fixed_jaw_link, jaw_box_cfg["fixed_jaw_box"])
        moving_box = get_jaw_box_world(moving_jaw_link, jaw_box_cfg["moving_jaw_box"])
        fixed_link_world = to_numpy(fixed_jaw_link.get_pos())
        moving_link_world = to_numpy(moving_jaw_link.get_pos())

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
        dist_to_fixed = float(np.dot(cube_center_world - fixed_inner_surface, fixed_inward))
        dist_to_moving = float(np.dot(cube_center_world - moving_inner_surface, moving_inward))

        return {
            "centering_error": float(np.linalg.norm(cube_center_world - jaw_midpoint)),
            "centering_xy": float(np.linalg.norm((cube_center_world - jaw_midpoint)[:2])),
            "jaw_balance_error": float(abs(dist_to_fixed - dist_to_moving)),
            "dist_to_fixed_jaw": dist_to_fixed,
            "dist_to_moving_jaw": dist_to_moving,
            "clearance_min": float(min(dist_to_fixed, dist_to_moving) - cube_half_extent),
            "jaw_gap": float(np.linalg.norm(moving_inner_surface - fixed_inner_surface)),
            "jaw_midpoint_world": jaw_midpoint.tolist(),
            "fixed_jaw_link_world": fixed_link_world.tolist(),
            "moving_jaw_link_world": moving_link_world.tolist(),
            "fixed_jaw_surface_world": fixed_inner_surface.tolist(),
            "moving_jaw_surface_world": moving_inner_surface.tolist(),
            "fixed_jaw_box_center_world": fixed_box["center_world"].tolist(),
            "moving_jaw_box_center_world": moving_box["center_world"].tolist(),
            "cube_center_world": cube_center_world.tolist(),
        }

    def solve_ik(pos, grip_deg, seed_rad=None):
        if seed_rad is None:
            seed_rad = home_rad
        so101.set_qpos(seed_rad)
        result = so101.inverse_kinematics(
            link=ee_link, pos=np.array(pos, dtype=np.float32),
            quat=np.array(IK_QUAT_DOWN, dtype=np.float32),
        )
        q = np.rad2deg(to_numpy(result))[:n_dofs].tolist()
        q[5] = grip_deg
        return q

    def lerp(a, b, n):
        a, b = np.array(a), np.array(b)
        return [a + (b - a) * (i + 1) / max(n, 1) for i in range(n)]

    def reset_scene(cpos, settle=30):
        so101.set_qpos(home_rad)
        so101.control_dofs_position(home_rad, dof_idx)
        so101.zero_all_dofs_velocity()
        cube.set_pos(torch.tensor(cpos, dtype=torch.float32, device=gs.device).unsqueeze(0))
        cube.set_quat(torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=gs.device).unsqueeze(0))
        cube.zero_all_dofs_velocity()
        for _ in range(settle):
            scene.step()

    cube_init = np.array([args.cube_x, args.cube_y, args.cube_z])

    def run_trial(ox, oy, oz):
        reset_scene(cube_init, settle=20)

        off = np.array([ox, oy, oz])
        pos_pre = cube_init + off + np.array([0, 0, 0.10])

        q_pre = solve_ik(pos_pre, args.gripper_open, seed_rad=home_rad)
        prev_rad = np.deg2rad(np.array(q_pre, dtype=np.float32))
        descent_wps = []
        for i in range(6):
            frac = (i + 1) / 6
            z = 0.10 + (args.approach_z - 0.10) * frac
            pos = cube_init + off + np.array([0, 0, z])
            wp = solve_ik(pos, args.gripper_open, seed_rad=prev_rad)
            descent_wps.append(wp)
            prev_rad = np.deg2rad(np.array(wp, dtype=np.float32))

        q_approach = descent_wps[-1]
        q_close = q_approach.copy()
        q_close[5] = args.gripper_close

        traj, phases = [], []
        n_move = max(15, args.trial_steps // 8)
        traj += lerp(HOME_DEG.tolist(), q_pre, n_move)
        phases += ["move_pre"] * n_move
        steps_per_wp = max(3, args.trial_steps // 48)
        prev = q_pre
        for wp in descent_wps:
            traj += lerp(prev, wp, steps_per_wp)
            phases += ["approach"] * steps_per_wp
            prev = wp
        n_close = max(8, args.trial_steps // 12)
        traj += lerp(q_approach, q_close, n_close)
        phases += ["close"] * n_close
        n_hold = 10
        traj += [q_close[:] for _ in range(n_hold)]
        phases += ["close_hold"] * n_hold

        approach_frames = []
        z_before = None
        z_after = None
        cube_pos_at_approach = None
        jaw_metrics_at_approach = None

        approach_indices = [i for i, ph in enumerate(phases) if ph == "approach"]
        approach_last10 = set(approach_indices[-10:]) if len(approach_indices) >= 10 else set(approach_indices)

        for fi, (target_deg, phase) in enumerate(zip(traj, phases)):
            target_rad = np.deg2rad(np.array(target_deg, dtype=np.float32))
            so101.control_dofs_position(target_rad, dof_idx)
            scene.step()

            z_now = float(to_numpy(cube.get_pos())[2])

            if fi in approach_last10:
                up_img = render_camera(cam_up)
                side_img = render_camera(cam_side)
                approach_frames.append((fi, np.concatenate([up_img, side_img], axis=1)))

            is_approach_last = (phase == "approach" and
                                (fi + 1 >= len(phases) or phases[fi + 1] != "approach"))
            if is_approach_last:
                cube_pos_at_approach = to_numpy(cube.get_pos())
                jaw_metrics_at_approach = compute_jaw_metrics(cube_pos_at_approach)

            if phase == "close" and z_before is None:
                z_before = z_now
            if phase == "close_hold":
                z_after = z_now

        if z_before is None:
            z_before = float(to_numpy(cube.get_pos())[2])
        if z_after is None:
            z_after = float(to_numpy(cube.get_pos())[2])

        delta_z = z_after - z_before

        if cube_pos_at_approach is not None and jaw_metrics_at_approach is not None:
            approach_contact = float(np.linalg.norm(cube_pos_at_approach - cube_init))
        else:
            approach_contact = 999.0
            jaw_metrics_at_approach = {
                "centering_error": 999.0,
                "centering_xy": 999.0,
                "jaw_balance_error": 999.0,
                "dist_to_fixed_jaw": -999.0,
                "dist_to_moving_jaw": -999.0,
                "clearance_min": -999.0,
                "jaw_gap": -999.0,
                "jaw_midpoint_world": [999.0, 999.0, 999.0],
                "fixed_jaw_link_world": [999.0, 999.0, 999.0],
                "moving_jaw_link_world": [999.0, 999.0, 999.0],
                "fixed_jaw_surface_world": [999.0, 999.0, 999.0],
                "moving_jaw_surface_world": [999.0, 999.0, 999.0],
                "fixed_jaw_box_center_world": [999.0, 999.0, 999.0],
                "moving_jaw_box_center_world": [999.0, 999.0, 999.0],
                "cube_center_world": [999.0, 999.0, 999.0],
            }

        return {
            "delta_z": delta_z,
            "approach_contact": approach_contact,
            "approach_frames": approach_frames,
            **jaw_metrics_at_approach,
        }

    # --- Run grid ---
    x_cands = parse_csv(args.x_candidates)
    y_cands = parse_csv(args.y_candidates)
    z_cands = parse_csv(args.z_candidates)
    total = len(x_cands) * len(y_cands) * len(z_cands)

    out_dir = Path(args.save) / args.exp_id
    png_dir = out_dir / "approach_pngs"
    png_dir.mkdir(parents=True, exist_ok=True)

    results = []
    tried = 0

    print(f"\nRunning {total} offset candidates...")
    print(
        f"  {'#':>4}  {'ox':>7} {'oy':>7} {'oz':>7}  "
        f"{'ctr3d':>7} {'bal':>7} {'d_fix':>7} {'d_mov':>7} "
        f"{'clr':>7} {'contact':>8}  flags"
    )
    print(f"  {'─'*100}")

    for oz in z_cands:
        for ox in x_cands:
            for oy in y_cands:
                tried += 1
                r = run_trial(ox, oy, oz)

                clean = r["approach_contact"] < args.contact_threshold
                flags = []
                if r["delta_z"] > 0.005:
                    flags.append("★dz")
                if clean:
                    flags.append("●clean")
                if r["clearance_min"] >= 0.0:
                    flags.append("□clear")
                flag_str = " ".join(flags) if flags else ""

                print(
                    f"  {tried:4d}  {ox:+.3f} {oy:+.3f} {oz:+.3f}  "
                    f"{r['centering_error']:.4f} {r['jaw_balance_error']:.4f} "
                    f"{r['dist_to_fixed_jaw']:.4f} {r['dist_to_moving_jaw']:.4f} "
                    f"{r['clearance_min']:+.4f} "
                    f"{r['approach_contact']:.5f}  {flag_str}"
                )

                for frame_idx, img in r["approach_frames"]:
                    save_png(img,
                             png_dir / f"ox{ox:+.3f}_oy{oy:+.3f}_oz{oz:+.3f}_f{frame_idx:03d}.png")

                results.append({
                    "ox": ox, "oy": oy, "oz": oz,
                    "delta_z": float(r["delta_z"]),
                    "centering_error": float(r["centering_error"]),
                    "centering_xy": float(r["centering_xy"]),
                    "jaw_balance_error": float(r["jaw_balance_error"]),
                    "dist_to_fixed_jaw": float(r["dist_to_fixed_jaw"]),
                    "dist_to_moving_jaw": float(r["dist_to_moving_jaw"]),
                    "clearance_min": float(r["clearance_min"]),
                    "jaw_gap": float(r["jaw_gap"]),
                    "approach_contact": float(r["approach_contact"]),
                    "jaw_midpoint_world": r["jaw_midpoint_world"],
                    "fixed_jaw_link_world": r["fixed_jaw_link_world"],
                    "moving_jaw_link_world": r["moving_jaw_link_world"],
                    "fixed_jaw_surface_world": r["fixed_jaw_surface_world"],
                    "moving_jaw_surface_world": r["moving_jaw_surface_world"],
                    "fixed_jaw_box_center_world": r["fixed_jaw_box_center_world"],
                    "moving_jaw_box_center_world": r["moving_jaw_box_center_world"],
                    "cube_center_world": r["cube_center_world"],
                })

    # --- Select best ---
    best_dz = max(results, key=lambda r: r["delta_z"])

    clean_results = [r for r in results if r["approach_contact"] < args.contact_threshold]
    clean_clear_results = [r for r in clean_results if r["clearance_min"] >= 0.0]
    if clean_clear_results:
        best_center = min(clean_clear_results, key=lambda r: r["centering_error"])
    elif clean_results:
        print("  Warning: no clean+clear candidate found, selecting from clean-only")
        best_center = min(clean_results, key=lambda r: r["centering_error"])
    else:
        print("  Warning: no clean approach found, selecting from all")
        best_center = min(results, key=lambda r: r["centering_error"])

    print(f"\n{'═'*70}")
    print(f"  RESULTS — {args.exp_id}")
    print(f"{'═'*70}")
    print(f"  Best by delta_z:")
    print(f"    offset=[{best_dz['ox']:+.3f}, {best_dz['oy']:+.3f}, {best_dz['oz']:+.3f}]")
    print(f"    delta_z={best_dz['delta_z']:+.4f}m")
    print(f"  Best by jaw midpoint centering:")
    print(f"    offset=[{best_center['ox']:+.3f}, {best_center['oy']:+.3f}, {best_center['oz']:+.3f}]")
    print(f"    centering_error={best_center['centering_error']:.4f}m")
    print(f"    jaw_balance_error={best_center['jaw_balance_error']:.4f}m")
    print(f"    dist_to_fixed_jaw={best_center['dist_to_fixed_jaw']:.4f}m")
    print(f"    dist_to_moving_jaw={best_center['dist_to_moving_jaw']:.4f}m")
    print(f"    clearance_min={best_center['clearance_min']:+.4f}m")
    print(f"    jaw_gap={best_center['jaw_gap']:.4f}m")
    print(f"    approach_contact={best_center['approach_contact']:.5f}m")
    print(f"  Approach PNGs: {png_dir}")
    print(f"  Clean candidates: {len(clean_results)}/{len(results)}")
    print(f"  Clean+clear candidates: {len(clean_clear_results)}/{len(results)}")
    print(f"{'═'*70}\n")

    with open(out_dir / "tune_results.json", "w") as f:
        json.dump({
            "results": results,
            "best_delta_z": best_dz,
            "best_centering": best_center,
            "contact_threshold": args.contact_threshold,
            "cube_half_extent": cube_half_extent,
            "jaw_box_config": {
                k: {
                    "body_name": v["body_name"],
                    "pos": v["pos"].tolist(),
                    "size": v["size"].tolist(),
                }
                for k, v in jaw_box_cfg.items()
            },
        }, f, indent=2)


if __name__ == "__main__":
    main()
