"""
SO-101 offset auto-tune with approach visualization.

Runs a grid of offset candidates, captures approach-end frame for each,
and selects best offset by either delta_z or approach centering.

Usage:
  python 10_auto_tune.py --xml assets/so101_new_calib_v3_jawbox.xml \
      --cube-friction 1.5 --save /output --exp-id T1_centering
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


def find_xml(user_path):
    if user_path:
        p = Path(user_path)
        if p.exists():
            return p
    for c in [Path("assets/so101_new_calib.xml"), Path("02_intermediate/scripts/assets/so101_new_calib.xml")]:
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


JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
HOME_DEG = np.array([0.0, -30.0, 90.0, -60.0, 0.0, 0.0], dtype=np.float32)
IK_QUAT_DOWN = np.array([0.0, 1.0, 0.0, 0.0])


def main():
    ensure_display()

    p = argparse.ArgumentParser(description="SO-101 offset auto-tune with approach visualization")
    p.add_argument("--xml", default=None)
    p.add_argument("--exp-id", default="T1")
    p.add_argument("--save", default="/output")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--gripper-open", type=float, default=15.0)
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
    args = p.parse_args()

    xml_path = find_xml(args.xml)
    if not xml_path:
        print("XML not found"); sys.exit(1)
    print(f"XML: {xml_path}")

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
        morph=gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=(args.cube_x, args.cube_y, args.cube_z)),
        surface=gs.surfaces.Default(color=(1.0, 0.3, 0.3, 1.0)),
    )
    if args.cube_friction is not None:
        cube_kw["material"] = gs.materials.Rigid(friction=args.cube_friction)
        print(f"  cube friction={args.cube_friction}")
    cube = scene.add_entity(**cube_kw)
    so101 = scene.add_entity(gs.morphs.MJCF(file=str(xml_path), pos=(0, 0, 0)))

    cam_up = scene.add_camera(res=(640, 480), pos=(0.42, 0.34, 0.26), lookat=(0.15, 0.0, 0.08), fov=38, GUI=False)
    cam_side = scene.add_camera(res=(640, 480), pos=(0.5, -0.4, 0.3), lookat=(0.15, 0.0, 0.1), fov=45, GUI=False)
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

    ee_link = so101.get_link("grasp_center")

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

    def reset_scene(cube_pos, settle=30):
        so101.set_qpos(home_rad)
        so101.control_dofs_position(home_rad, dof_idx)
        so101.zero_all_dofs_velocity()
        cube.set_pos(torch.tensor(cube_pos, dtype=torch.float32, device=gs.device).unsqueeze(0))
        cube.set_quat(torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=gs.device).unsqueeze(0))
        cube.zero_all_dofs_velocity()
        for _ in range(settle):
            scene.step()

    cube_pos = np.array([args.cube_x, args.cube_y, args.cube_z])

    def run_trial_with_snapshot(ox, oy, oz):
        """Run trial, capture approach-end frame, return (delta_z, approach_img)."""
        reset_scene(cube_pos, settle=20)

        off = np.array([ox, oy, oz])
        pos_pre = cube_pos + off + np.array([0, 0, 0.10])

        q_pre = solve_ik(pos_pre, args.gripper_open, seed_rad=home_rad)
        prev_rad = np.deg2rad(np.array(q_pre, dtype=np.float32))
        descent_wps = []
        for i in range(6):
            frac = (i + 1) / 6
            z = 0.10 + (args.approach_z - 0.10) * frac
            pos = cube_pos + off + np.array([0, 0, z])
            wp = solve_ik(pos, args.gripper_open, seed_rad=prev_rad)
            descent_wps.append(wp)
            prev_rad = np.deg2rad(np.array(wp, dtype=np.float32))

        q_approach = descent_wps[-1]
        q_close = q_approach.copy()
        q_close[5] = args.gripper_close

        traj = []
        phases = []
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

        approach_img = None
        z_before = None
        z_after = None
        cube_z_at_approach_end = None

        for fi, (target_deg, phase) in enumerate(zip(traj, phases)):
            target_rad = np.deg2rad(np.array(target_deg, dtype=np.float32))
            so101.control_dofs_position(target_rad, dof_idx)
            scene.step()

            z_now = float(to_numpy(cube.get_pos())[2])

            if phase == "approach" and (fi + 1 >= len(phases) or phases[fi + 1] != "approach"):
                up_img = render_camera(cam_up)
                side_img = render_camera(cam_side)
                approach_img = np.concatenate([up_img, side_img], axis=1)
                cube_z_at_approach_end = z_now

            if phase == "close" and z_before is None:
                z_before = z_now
            if phase == "close_hold":
                z_after = z_now

        if z_before is None:
            z_before = float(to_numpy(cube.get_pos())[2])
        if z_after is None:
            z_after = float(to_numpy(cube.get_pos())[2])

        delta_z = z_after - z_before
        approach_delta = abs(cube_z_at_approach_end - args.cube_z) if cube_z_at_approach_end else 999

        return {
            "delta_z": delta_z,
            "approach_delta": approach_delta,
            "approach_img": approach_img,
        }

    # Run grid
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
    for oz in z_cands:
        for ox in x_cands:
            for oy in y_cands:
                tried += 1
                r = run_trial_with_snapshot(ox, oy, oz)

                tag_dz = "★" if r["delta_z"] > 0.005 else " "
                tag_ap = "●" if r["approach_delta"] < 0.002 else " "
                print(
                    f"  {tag_dz}{tag_ap} [{tried}/{total}] "
                    f"ox={ox:+.3f} oy={oy:+.3f} oz={oz:+.3f} "
                    f"→ Δz={r['delta_z']:+.4f}m  approach_Δ={r['approach_delta']:.4f}m"
                )

                if r["approach_img"] is not None:
                    fname = f"ox{ox:+.3f}_oy{oy:+.3f}_oz{oz:+.3f}.png"
                    save_png(r["approach_img"], png_dir / fname)

                results.append({
                    "ox": ox, "oy": oy, "oz": oz,
                    "delta_z": float(r["delta_z"]),
                    "approach_delta": float(r["approach_delta"]),
                })

    # Select best by delta_z
    best_dz = max(results, key=lambda r: r["delta_z"])
    # Select best by approach centering (smallest approach_delta = least cube disturbance)
    best_ap = min(results, key=lambda r: r["approach_delta"])

    print(f"\n{'═'*60}")
    print(f"  RESULTS — {args.exp_id}")
    print(f"{'═'*60}")
    print(f"  Best by delta_z:     ox={best_dz['ox']:+.3f} oy={best_dz['oy']:+.3f} oz={best_dz['oz']:+.3f}  Δz={best_dz['delta_z']:+.4f}m")
    print(f"  Best by centering:   ox={best_ap['ox']:+.3f} oy={best_ap['oy']:+.3f} oz={best_ap['oz']:+.3f}  approach_Δ={best_ap['approach_delta']:.4f}m")
    print(f"  Approach PNGs: {png_dir}")
    print(f"{'═'*60}\n")

    with open(out_dir / "tune_results.json", "w") as f:
        json.dump({"results": results, "best_delta_z": best_dz, "best_centering": best_ap}, f, indent=2)


if __name__ == "__main__":
    main()
