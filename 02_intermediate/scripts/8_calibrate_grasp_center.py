"""
Calibrate SO-101 grasp_center against the actual jaw pinch center.

Goal:
- Run a deterministic fixed-pose XY offset sweep with the current `grasp_center`
- Find the runtime world-frame offset that best centers the cube between jaws
- Convert that runtime world-frame offset into a suggested XML local-frame update

Why this works:
- Today, IK targets `grasp_center`
- If the best runtime grasp needs an offset `(ox, oy, oz)`, that means the current
  `grasp_center` body is displaced from the real jaw pinch center
- We can convert that best world-frame offset back into gripper-local coordinates
  and suggest a new `grasp_center.pos` for the MJCF asset

This script prioritizes transparency over over-automation:
- It prints coarse and refined search logs
- It exports stitched close-phase PNGs for the top-K candidates
- It writes a calibration summary with:
  - best runtime offset in world frame
  - recommended local-frame delta for `grasp_center.pos`
  - recommended new `grasp_center.pos`
"""

import argparse
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
    else:
        print("[display] WARNING: Xvfb exited immediately")


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


def parse_xml_grasp_center(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    grasp_center = None
    for body in root.iter("body"):
        if body.attrib.get("name") == "grasp_center":
            grasp_center = body
            break
    if grasp_center is None:
        raise RuntimeError("grasp_center body not found in XML")
    pos = np.array([float(v) for v in grasp_center.attrib["pos"].split()], dtype=np.float64)
    quat = np.array([float(v) for v in grasp_center.attrib["quat"].split()], dtype=np.float64)
    return pos, quat


def build_symmetric_grid(center, radius, step):
    start = center - radius
    end = center + radius
    n = int(round((end - start) / step))
    return [round(start + i * step, 6) for i in range(n + 1)]


def rank_key(result):
    # Primary: lift_delta
    # Secondary: close_contact_delta
    # Tertiary: minimize XY push (less pushing, more likely centered)
    return (
        result["lift_delta"],
        result["close_contact_delta"],
        -result["xy_push"],
    )


def main():
    ensure_display()

    parser = argparse.ArgumentParser(description="Calibrate SO-101 grasp_center local pos")
    parser.add_argument("--xml", required=True, help="Path to so101_new_calib.xml")
    parser.add_argument("--exp-id", default="C1_gc_calibration")
    parser.add_argument("--save", default="/output")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--img-w", type=int, default=640)
    parser.add_argument("--img-h", type=int, default=480)
    parser.add_argument("--cube-x", type=float, default=0.16)
    parser.add_argument("--cube-y", type=float, default=0.0)
    parser.add_argument("--cube-z", type=float, default=0.015)
    parser.add_argument("--approach-z", type=float, default=0.012)
    parser.add_argument("--gripper-open", type=float, default=0.0)
    parser.add_argument("--gripper-close", type=float, default=-20.0)
    parser.add_argument("--close-hold-steps", type=int, default=50)
    parser.add_argument("--coarse-radius", type=float, default=0.02)
    parser.add_argument("--coarse-step", type=float, default=0.004)
    parser.add_argument("--refine-radius", type=float, default=0.004)
    parser.add_argument("--refine-step", type=float, default=0.001)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    xml_path = Path(args.xml)
    gc_local_pos, gc_local_quat = parse_xml_grasp_center(xml_path)
    print(
        "Current XML grasp_center:"
        f" pos={gc_local_pos.tolist()} quat={gc_local_quat.tolist()}"
    )

    import torch
    import genesis as gs
    import genesis.utils.geom as gu

    JOINT_NAMES = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]
    HOME_DEG = np.array([0.0, -30.0, 90.0, -60.0, 0.0, 0.0], dtype=np.float32)
    KP = np.array([500.0, 500.0, 400.0, 400.0, 300.0, 200.0])
    KV = np.array([50.0, 50.0, 40.0, 40.0, 30.0, 20.0])

    backend = gs.cpu if args.cpu else gs.gpu
    gs.init(backend=backend, logging_level="warning")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / args.fps, substeps=4),
        rigid_options=gs.options.RigidOptions(enable_collision=True, enable_joint_limit=True),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    cube = scene.add_entity(
        gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=(args.cube_x, args.cube_y, args.cube_z)),
        surface=gs.surfaces.Default(color=(1.0, 0.3, 0.3, 1.0)),
    )
    so101 = scene.add_entity(gs.morphs.MJCF(file=str(xml_path), pos=(0.0, 0.0, 0.0)))

    cam_up = scene.add_camera(
        res=(args.img_w, args.img_h),
        pos=(0.42, 0.34, 0.26),
        lookat=(args.cube_x, args.cube_y, 0.08),
        fov=38,
        GUI=False,
    )
    cam_side = scene.add_camera(
        res=(args.img_w, args.img_h),
        pos=(0.5, -0.4, 0.3),
        lookat=(args.cube_x, args.cube_y, 0.1),
        fov=45,
        GUI=False,
    )
    scene.build()

    n_dofs = so101.n_dofs
    dof_idx = np.arange(n_dofs)
    so101.set_dofs_kp(KP[:n_dofs], dof_idx)
    so101.set_dofs_kv(KV[:n_dofs], dof_idx)
    home_deg = HOME_DEG[:n_dofs]
    home_rad = np.deg2rad(home_deg)
    so101.set_qpos(home_rad)
    so101.control_dofs_position(home_rad, dof_idx)
    for _ in range(60):
        scene.step()

    ee_link = so101.get_link("grasp_center")
    gripper_link = so101.get_link("gripper")
    moving_jaw_link = so101.get_link("moving_jaw_so101_v1")

    def reset_scene(cube_pos, settle_steps=20):
        so101.set_qpos(home_rad)
        so101.control_dofs_position(home_rad, dof_idx)
        so101.zero_all_dofs_velocity()
        cube.set_pos(torch.tensor(cube_pos, dtype=torch.float32, device=gs.device).unsqueeze(0))
        cube.set_quat(
            torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=gs.device).unsqueeze(0)
        )
        cube.zero_all_dofs_velocity()
        for _ in range(settle_steps):
            scene.step()

    def solve_ik_seeded(pos, grip_deg, seed_rad=None):
        q = to_numpy(
            so101.inverse_kinematics(
                link=ee_link,
                pos=pos,
                quat=None,
                init_qpos=seed_rad,
                max_solver_iters=50,
                damping=0.02,
            )
        )
        q_deg = np.rad2deg(q)
        if n_dofs >= 6:
            q_deg[5] = grip_deg
        return q_deg

    def lerp(a, b, n):
        return [a + (b - a) * (i + 1) / max(n, 1) for i in range(n)]

    def build_trajectory(cube_pos, ox, oy, oz=0.0, total_steps=90):
        off = np.array([ox, oy, oz], dtype=np.float64)
        pos_pre = cube_pos + off + np.array([0.0, 0.0, 0.10])
        pos_approach = cube_pos + off + np.array([0.0, 0.0, args.approach_z])

        q_pre = solve_ik_seeded(pos_pre, args.gripper_open, seed_rad=home_rad)
        q_pre_rad = np.deg2rad(np.array(q_pre, dtype=np.float32))

        n_descent_wps = 6
        descent_wps = []
        prev_rad = q_pre_rad
        for i in range(n_descent_wps):
            frac = (i + 1) / n_descent_wps
            z = 0.10 + (args.approach_z - 0.10) * frac
            pos = cube_pos + off + np.array([0.0, 0.0, z])
            wp = solve_ik_seeded(pos, args.gripper_open, seed_rad=prev_rad)
            descent_wps.append(wp)
            prev_rad = np.deg2rad(np.array(wp, dtype=np.float32))

        q_approach = descent_wps[-1]
        q_close = q_approach.copy()
        if n_dofs >= 6:
            q_close[5] = args.gripper_close

        q_close_rad = np.deg2rad(np.array(q_close, dtype=np.float32))
        n_lift_wps = 4
        lift_wps = []
        prev_rad = q_close_rad
        for i in range(n_lift_wps):
            frac = (i + 1) / n_lift_wps
            z = args.approach_z + (0.15 - args.approach_z) * frac
            pos = cube_pos + off + np.array([0.0, 0.0, z])
            wp = solve_ik_seeded(pos, args.gripper_close, seed_rad=prev_rad)
            lift_wps.append(wp)
            prev_rad = np.deg2rad(np.array(wp, dtype=np.float32))

        q_lift = lift_wps[-1]
        traj, phases = [], []

        n_move = max(15, total_steps // 8)
        traj += lerp(home_deg.copy(), q_pre, n_move)
        phases += ["move_pre"] * n_move

        steps_per_wp = max(3, total_steps // (8 * n_descent_wps))
        prev = q_pre
        for wp in descent_wps:
            traj += lerp(prev, wp, steps_per_wp)
            phases += ["approach"] * steps_per_wp
            prev = wp

        n_close = max(8, total_steps // 12)
        traj += lerp(q_approach, q_close, n_close)
        phases += ["close"] * n_close

        traj += [q_close.copy() for _ in range(args.close_hold_steps)]
        phases += ["close_hold"] * args.close_hold_steps

        steps_per_lift = max(5, total_steps // (8 * n_lift_wps))
        prev = q_close
        for wp in lift_wps:
            traj += lerp(prev, wp, steps_per_lift)
            phases += ["lift"] * steps_per_lift
            prev = wp

        consumed = len(traj)
        n_return = max(0, total_steps - consumed)
        if n_return > 0:
            traj += lerp(q_lift, home_deg.copy(), n_return)
            phases += ["return"] * n_return
        return traj, phases

    def evaluate_candidate(cube_pos, ox, oy, oz=0.0, save_preview=False, preview_path=None):
        reset_scene(cube_pos, settle_steps=20)
        traj, phases = build_trajectory(cube_pos, ox, oy, oz, total_steps=90)

        z_before_close = None
        z_after_lift = None
        z_peak_close_hold = -1e9
        cube_xy_before_close = None
        cube_xy_end_hold = None
        close_frame = None
        close_frame_idx = None
        close_gripper_pos = None
        close_gripper_quat = None
        close_cube_local_gripper = None
        close_gc_world = None
        close_mj_world = None

        for frame_idx, (target_deg, phase) in enumerate(zip(traj, phases)):
            target_rad = np.deg2rad(np.array(target_deg, dtype=np.float32))
            so101.control_dofs_position(target_rad, dof_idx)
            scene.step()

            cube_pos_now = to_numpy(cube.get_pos())
            z_now = float(cube_pos_now[2])

            if phase == "close" and z_before_close is None:
                z_before_close = z_now
                cube_xy_before_close = cube_pos_now[:2].copy()
                gripper_pos = to_numpy(gripper_link.get_pos())
                gripper_quat = to_numpy(gripper_link.get_quat())
                close_gripper_pos = gripper_pos.copy()
                close_gripper_quat = gripper_quat.copy()
                close_cube_local_gripper = gu.inv_transform_by_quat(
                    cube_pos_now - gripper_pos, gripper_quat
                )
                close_gc_world = to_numpy(ee_link.get_pos())
                close_mj_world = to_numpy(moving_jaw_link.get_pos())
                if save_preview or preview_path is not None:
                    stitched = np.concatenate(
                        [render_camera(cam_up), render_camera(cam_side)], axis=1
                    )
                    close_frame = stitched
                    close_frame_idx = frame_idx

            if phase in {"close", "close_hold"}:
                z_peak_close_hold = max(z_peak_close_hold, z_now)

            if phase == "close_hold":
                cube_xy_end_hold = cube_pos_now[:2].copy()

            if phase == "lift":
                z_after_lift = z_now

        if z_before_close is None:
            z_before_close = float(to_numpy(cube.get_pos())[2])
        if z_after_lift is None:
            z_after_lift = float(to_numpy(cube.get_pos())[2])
        if cube_xy_before_close is None:
            cube_xy_before_close = to_numpy(cube.get_pos())[:2].copy()
        if cube_xy_end_hold is None:
            cube_xy_end_hold = to_numpy(cube.get_pos())[:2].copy()

        lift_delta = z_after_lift - z_before_close
        close_contact_delta = z_peak_close_hold - z_before_close
        xy_push = float(np.linalg.norm(cube_xy_end_hold - cube_xy_before_close))

        world_offset = np.array([ox, oy, oz], dtype=np.float64)
        local_delta = None
        suggested_local_pos = None
        if close_gripper_quat is not None:
            local_delta = -gu.inv_transform_by_quat(world_offset, close_gripper_quat)
            suggested_local_pos = gc_local_pos + local_delta

        result = {
            "offset_world": [float(ox), float(oy), float(oz)],
            "lift_delta": float(lift_delta),
            "close_contact_delta": float(close_contact_delta),
            "xy_push": float(xy_push),
            "z_before_close": float(z_before_close),
            "z_after_lift": float(z_after_lift),
            "cube_local_in_gripper_at_close": close_cube_local_gripper.tolist()
            if close_cube_local_gripper is not None
            else None,
            "gripper_world_at_close": close_gripper_pos.tolist()
            if close_gripper_pos is not None
            else None,
            "gripper_quat_at_close": close_gripper_quat.tolist()
            if close_gripper_quat is not None
            else None,
            "grasp_center_world_at_close": close_gc_world.tolist()
            if close_gc_world is not None
            else None,
            "moving_jaw_world_at_close": close_mj_world.tolist()
            if close_mj_world is not None
            else None,
            "suggested_grasp_center_delta_local": local_delta.tolist()
            if local_delta is not None
            else None,
            "suggested_grasp_center_pos_local": suggested_local_pos.tolist()
            if suggested_local_pos is not None
            else None,
            "close_frame_index": close_frame_idx,
        }

        if save_preview and close_frame is not None and preview_path is not None:
            save_rgb_png(close_frame, preview_path)

        return result

    cube_pos = np.array([args.cube_x, args.cube_y, args.cube_z], dtype=np.float64)
    out_dir = Path(args.save) / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "topk_close_pngs"
    preview_dir.mkdir(parents=True, exist_ok=True)

    # Coarse search
    print("\n" + "=" * 70)
    print("COARSE SEARCH")
    print("=" * 70)
    coarse_x = build_symmetric_grid(0.0, args.coarse_radius, args.coarse_step)
    coarse_y = build_symmetric_grid(0.0, args.coarse_radius, args.coarse_step)
    coarse_results = []
    total = len(coarse_x) * len(coarse_y)
    tried = 0
    for ox in coarse_x:
        for oy in coarse_y:
            tried += 1
            result = evaluate_candidate(cube_pos, ox, oy, oz=0.0)
            coarse_results.append(result)
            print(
                f"[coarse {tried:03d}/{total}] ox={ox:+.3f} oy={oy:+.3f} "
                f"lift={result['lift_delta']:+.4f} "
                f"contact={result['close_contact_delta']:+.4f} "
                f"xy_push={result['xy_push']:.4f}"
            )

    coarse_results.sort(key=rank_key, reverse=True)
    coarse_best = coarse_results[0]
    coarse_best_ox, coarse_best_oy, _ = coarse_best["offset_world"]
    print("\nBest coarse candidate:")
    print(json.dumps(coarse_best, indent=2))

    # Refine search around coarse best
    print("\n" + "=" * 70)
    print("REFINE SEARCH")
    print("=" * 70)
    refine_x = build_symmetric_grid(coarse_best_ox, args.refine_radius, args.refine_step)
    refine_y = build_symmetric_grid(coarse_best_oy, args.refine_radius, args.refine_step)
    refine_results = []
    total = len(refine_x) * len(refine_y)
    tried = 0
    for ox in refine_x:
        for oy in refine_y:
            tried += 1
            result = evaluate_candidate(cube_pos, ox, oy, oz=0.0)
            refine_results.append(result)
            print(
                f"[refine {tried:03d}/{total}] ox={ox:+.4f} oy={oy:+.4f} "
                f"lift={result['lift_delta']:+.4f} "
                f"contact={result['close_contact_delta']:+.4f} "
                f"xy_push={result['xy_push']:.4f}"
            )

    refine_results.sort(key=rank_key, reverse=True)
    best = refine_results[0]

    # Export top-K preview frames for manual review
    top_k = refine_results[: args.top_k]
    for rank, result in enumerate(top_k, start=1):
        ox, oy, oz = result["offset_world"]
        filename = (
            f"rank{rank:02d}_ox{ox:+.4f}_oy{oy:+.4f}"
            f"_lift{result['lift_delta']:+.4f}_push{result['xy_push']:.4f}.png"
        )
        preview_path = preview_dir / filename
        evaluate_candidate(cube_pos, ox, oy, oz=oz, save_preview=True, preview_path=preview_path)

    summary = {
        "exp_id": args.exp_id,
        "xml_path": str(xml_path),
        "cube_pos": cube_pos.tolist(),
        "approach_z": args.approach_z,
        "gripper_open_deg": args.gripper_open,
        "gripper_close_deg": args.gripper_close,
        "close_hold_steps": args.close_hold_steps,
        "current_grasp_center_pos_local": gc_local_pos.tolist(),
        "current_grasp_center_quat_local": gc_local_quat.tolist(),
        "coarse_best": coarse_best,
        "refine_best": best,
        "top_k": top_k,
    }

    with open(out_dir / "calibration_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)
    print(f"Current grasp_center.pos (local): {gc_local_pos.tolist()}")
    print(f"Best runtime offset (world):      {best['offset_world']}")
    print(f"Suggested local delta:            {best['suggested_grasp_center_delta_local']}")
    print(f"Suggested new grasp_center.pos:   {best['suggested_grasp_center_pos_local']}")
    print(f"Best lift/contact/push:           "
          f"{best['lift_delta']:+.4f} / {best['close_contact_delta']:+.4f} / {best['xy_push']:.4f}")
    print(f"Top-K preview PNGs:               {preview_dir}")
    print(f"Summary JSON:                     {out_dir / 'calibration_summary.json'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
