"""
SO-101 oriented grasp test — enforce downward orientation for the gripper link.

Insight from Genesis Franka example: the hand link is targeted ABOVE the cube
with quat=[0,1,0,0] (pointing down), so the fingers naturally hang below and
close around the cube.

For SO-101, we target the `gripper` link (not grasp_center) with downward
orientation, placing it above the cube so the jaw mechanism hangs at cube level.

Usage:
  python 6_oriented_grasp_test.py --platform-height 0.06
  python 6_oriented_grasp_test.py --platform-height 0.06 --sweep-height
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
    subprocess.Popen(
        ["Xvfb", ":99", "-screen", "0", "1280x1024x24", "-ac", "+extension", "GLX"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    os.environ["DISPLAY"] = ":99"
    time.sleep(2)


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


HOME_DEG = np.array([0.0, -30.0, 90.0, -60.0, 0.0, 0.0], dtype=np.float32)
KP = np.array([500.0, 500.0, 400.0, 400.0, 300.0, 200.0])
KV = np.array([50.0, 50.0, 40.0, 40.0, 30.0, 20.0])
IK_QUAT_DOWN = np.array([0.0, 1.0, 0.0, 0.0])


def main():
    ensure_display()
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=None)
    ap.add_argument("--exp-id", default="O1_oriented")
    ap.add_argument("--save", default="/output")
    ap.add_argument("--platform-height", type=float, default=0.06)
    ap.add_argument("--cube-size", type=float, default=0.03)
    ap.add_argument("--cube-x", type=float, default=0.16)
    ap.add_argument("--cube-y", type=float, default=0.0)
    ap.add_argument("--gripper-height-above-cube", type=float, default=0.05,
                    help="Height of gripper link above cube center (jaws hang below)")
    ap.add_argument("--gripper-open", type=float, default=0.0)
    ap.add_argument("--gripper-close", type=float, default=-20.0)
    ap.add_argument("--close-hold-steps", type=int, default=60)
    ap.add_argument("--lift-threshold", type=float, default=0.01)
    ap.add_argument("--sweep-height", action="store_true",
                    help="Sweep gripper-height-above-cube to find optimal")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--episode-length", type=float, default=10.0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    xml_path = args.xml
    if xml_path is None:
        for p in [
            Path("/workspace/lfzte/02_intermediate/scripts/assets/so101_new_calib.xml"),
            Path("assets/so101_new_calib.xml"),
        ]:
            if p.exists():
                xml_path = str(p)
                break
    if xml_path is None:
        try:
            from huggingface_hub import snapshot_download
            d = snapshot_download(repo_type="dataset", repo_id="Genesis-Intelligence/assets",
                                 allow_patterns="SO101/*", max_workers=1)
            xml_path = str(Path(d) / "SO101" / "so101_new_calib.xml")
        except Exception as e:
            print(f"MJCF not found: {e}")
            sys.exit(1)
    print(f"XML: {xml_path}")

    import torch
    import genesis as gs

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

    ph = args.platform_height
    if ph > 0:
        scene.add_entity(
            gs.morphs.Box(size=(0.10, 0.10, ph), pos=(args.cube_x, args.cube_y, ph / 2), fixed=True),
            surface=gs.surfaces.Default(color=(0.6, 0.6, 0.6, 1.0)),
        )
    cs = args.cube_size
    cube_z = (ph + cs / 2) if ph > 0 else cs / 2
    cube = scene.add_entity(
        gs.morphs.Box(size=(cs, cs, cs), pos=(args.cube_x, args.cube_y, cube_z)),
        surface=gs.surfaces.Default(color=(1.0, 0.3, 0.3, 1.0)),
    )
    so101 = scene.add_entity(gs.morphs.MJCF(file=xml_path, pos=(0, 0, 0)))

    cam_up = scene.add_camera(
        res=(640, 480),
        pos=(0.42, 0.34, 0.26), lookat=(args.cube_x, args.cube_y, cube_z + 0.05), fov=38, GUI=False,
    )
    cam_side = scene.add_camera(
        res=(640, 480),
        pos=(0.5, -0.4, 0.3), lookat=(args.cube_x, args.cube_y, cube_z + 0.05), fov=45, GUI=False,
    )
    scene.build()
    print(f"Scene: platform_h={ph}, cube_center_z={cube_z:.4f}")

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

    gripper_link = so101.get_link("gripper")
    try:
        gc_link = so101.get_link("grasp_center")
    except Exception:
        gc_link = None
    try:
        mj_link = so101.get_link("moving_jaw_so101_v1")
    except Exception:
        mj_link = None

    print(f"Links: gripper=OK, grasp_center={'OK' if gc_link else 'N/A'}, moving_jaw={'OK' if mj_link else 'N/A'}")

    # Test IK with orientation constraint
    test_target = np.array([args.cube_x, args.cube_y, cube_z + 0.06])
    print(f"\n--- IK with quat=[0,1,0,0] (gripper down), target={test_target.tolist()} ---")
    try:
        q_test = so101.inverse_kinematics(
            link=gripper_link, pos=test_target, quat=IK_QUAT_DOWN,
            init_qpos=home_rad, max_solver_iters=100, damping=0.05,
        )
        q_np = to_numpy(q_test)
        so101.set_qpos(q_np)
        so101.control_dofs_position(q_np, dof_idx)
        for _ in range(20):
            scene.step()
        grip_pos = to_numpy(gripper_link.get_pos())
        ik_err = np.linalg.norm(grip_pos - test_target)
        print(f"  gripper actual: [{grip_pos[0]:.4f}, {grip_pos[1]:.4f}, {grip_pos[2]:.4f}], err={ik_err:.4f}")
        for link in so101.links:
            p = to_numpy(link.get_pos())
            print(f"    {link.name:30s} [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]")
        ik_with_quat = True
    except Exception as e:
        print(f"  IK with quat failed: {e}")
        ik_with_quat = False

    print(f"\n--- IK without quat (gripper free orient), target={test_target.tolist()} ---")
    q_test2 = so101.inverse_kinematics(
        link=gripper_link, pos=test_target, quat=None,
        init_qpos=home_rad, max_solver_iters=100, damping=0.05,
    )
    q_np2 = to_numpy(q_test2)
    so101.set_qpos(q_np2)
    so101.control_dofs_position(q_np2, dof_idx)
    for _ in range(20):
        scene.step()
    grip_pos2 = to_numpy(gripper_link.get_pos())
    ik_err2 = np.linalg.norm(grip_pos2 - test_target)
    print(f"  gripper actual: [{grip_pos2[0]:.4f}, {grip_pos2[1]:.4f}, {grip_pos2[2]:.4f}], err={ik_err2:.4f}")
    for link in so101.links:
        p = to_numpy(link.get_pos())
        print(f"    {link.name:30s} [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]")

    # Determine which IK mode to use
    use_quat = ik_with_quat
    if ik_with_quat:
        # Check if quat-constrained IK is better
        if ik_err > 0.03:
            print(f"\n  quat-constrained IK err={ik_err:.4f} > 0.03 — falling back to free orientation")
            use_quat = False

    print(f"\nUsing IK {'WITH' if use_quat else 'WITHOUT'} orientation constraint")

    so101.set_qpos(home_rad)
    for _ in range(30):
        scene.step()

    def solve_ik(pos, grip_deg, seed_rad=None):
        quat = IK_QUAT_DOWN if use_quat else None
        q = to_numpy(so101.inverse_kinematics(
            link=gripper_link, pos=pos, quat=quat,
            init_qpos=seed_rad, max_solver_iters=100, damping=0.05,
        ))
        q_deg = np.rad2deg(q)
        if n_dofs >= 6:
            q_deg[5] = grip_deg
        return q_deg

    def lerp(a, b, n):
        return [a + (b - a) * (i + 1) / max(n, 1) for i in range(n)]

    def reset_scene(settle=30):
        so101.set_qpos(home_rad)
        so101.control_dofs_position(home_rad, dof_idx)
        so101.zero_all_dofs_velocity()
        cpos = np.array([args.cube_x, args.cube_y, cube_z])
        cube.set_pos(torch.tensor(cpos, dtype=torch.float32, device=gs.device).unsqueeze(0))
        cube.set_quat(torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=gs.device).unsqueeze(0))
        cube.zero_all_dofs_velocity()
        for _ in range(settle):
            scene.step()

    def build_traj(cube_pos, h_above, total_steps):
        """Straight top-down approach: gripper link positioned above the cube, jaws hang below."""
        grasp_z = cube_pos[2] + h_above
        pre_z = cube_pos[2] + 0.12
        lift_z = cube_pos[2] + 0.12

        pos_pre = np.array([cube_pos[0], cube_pos[1], pre_z])
        pos_grasp = np.array([cube_pos[0], cube_pos[1], grasp_z])
        pos_lift = np.array([cube_pos[0], cube_pos[1], lift_z])

        q_pre = solve_ik(pos_pre, args.gripper_open, seed_rad=home_rad)
        prev_rad = np.deg2rad(np.array(q_pre, dtype=np.float32))

        traj, phases = [], []

        # Move to pre-position
        n_move = max(15, total_steps // 8)
        traj += lerp(home_deg.copy(), q_pre, n_move)
        phases += ["move_pre"] * n_move

        # Descend to grasp height
        n_desc = 6
        prev = q_pre
        for i in range(n_desc):
            frac = (i + 1) / n_desc
            z = pre_z + (grasp_z - pre_z) * frac
            p = np.array([cube_pos[0], cube_pos[1], z])
            wp = solve_ik(p, args.gripper_open, seed_rad=prev_rad)
            spw = max(4, total_steps // (8 * n_desc))
            traj += lerp(prev, wp, spw)
            phases += ["descend"] * spw
            prev = wp
            prev_rad = np.deg2rad(np.array(wp, dtype=np.float32))

        # Close
        q_approach = prev
        q_close = q_approach.copy()
        if n_dofs >= 6:
            q_close[5] = args.gripper_close
        n_close = max(10, total_steps // 12)
        traj += lerp(q_approach, q_close, n_close)
        phases += ["close"] * n_close

        # Hold
        n_hold = args.close_hold_steps
        traj += [q_close.copy() for _ in range(n_hold)]
        phases += ["close_hold"] * n_hold

        # Lift
        q_close_rad = np.deg2rad(np.array(q_close, dtype=np.float32))
        n_lift_wps = 4
        prev = q_close
        prev_rad = q_close_rad
        for i in range(n_lift_wps):
            frac = (i + 1) / n_lift_wps
            z = grasp_z + (lift_z - grasp_z) * frac
            p = np.array([cube_pos[0], cube_pos[1], z])
            wp = solve_ik(p, args.gripper_close, seed_rad=prev_rad)
            slw = max(5, total_steps // (8 * n_lift_wps))
            traj += lerp(prev, wp, slw)
            phases += ["lift"] * slw
            prev = wp
            prev_rad = np.deg2rad(np.array(wp, dtype=np.float32))

        consumed = len(traj)
        n_ret = max(0, total_steps - consumed)
        if n_ret > 0:
            traj += lerp(prev, home_deg.copy(), n_ret)
            phases += ["return"] * n_ret
        return traj, phases

    def run_trial(cube_pos, h_above):
        reset_scene(settle=20)
        traj, phases = build_traj(cube_pos, h_above, total_steps=120)
        z_before = z_after = None
        for td, phase in zip(traj, phases):
            tr = np.deg2rad(np.array(td, dtype=np.float32))
            so101.control_dofs_position(tr, dof_idx)
            scene.step()
            z = float(to_numpy(cube.get_pos())[2])
            if phase == "close" and z_before is None:
                z_before = z
            if phase == "lift":
                z_after = z
        if z_before is None:
            z_before = float(to_numpy(cube.get_pos())[2])
        if z_after is None:
            z_after = float(to_numpy(cube.get_pos())[2])
        return z_after - z_before

    import torch

    reset_scene()
    cube_pos_arr = to_numpy(cube.get_pos())
    print(f"\ncube settled at [{cube_pos_arr[0]:.4f}, {cube_pos_arr[1]:.4f}, {cube_pos_arr[2]:.4f}]")

    if args.sweep_height:
        print("\n=== Height sweep ===")
        print(f"{'h_above':>10} {'grip_z':>10} {'mj_z':>10} {'delta_z':>10} {'note':>10}")
        print("-" * 60)
        best_h = args.gripper_height_above_cube
        best_dz = -1e9
        for h in np.arange(0.02, 0.12, 0.005):
            reset_scene(settle=10)
            grasp_target = np.array([cube_pos_arr[0], cube_pos_arr[1], cube_pos_arr[2] + h])
            q = to_numpy(so101.inverse_kinematics(
                link=gripper_link, pos=grasp_target,
                quat=IK_QUAT_DOWN if use_quat else None,
                init_qpos=home_rad, max_solver_iters=100, damping=0.05,
            ))
            so101.set_qpos(q)
            for _ in range(10):
                scene.step()
            gp = to_numpy(gripper_link.get_pos())
            mp = to_numpy(mj_link.get_pos()) if mj_link else gp

            try:
                dz = run_trial(cube_pos_arr, h)
            except Exception:
                dz = -1.0

            tag = "***" if dz > args.lift_threshold else ""
            print(f"{h:10.3f} {gp[2]:10.4f} {mp[2]:10.4f} {dz:10.4f} {tag:>10}")
            if dz > best_dz:
                best_dz = dz
                best_h = h
        print(f"\nbest h_above = {best_h:.3f}, delta_z = {best_dz:.4f}")
        args.gripper_height_above_cube = best_h

    # Full episode
    print(f"\n=== Full episode: h_above={args.gripper_height_above_cube:.3f} ===")
    steps_per_ep = int(args.episode_length * args.fps)
    reset_scene(settle=30)
    cube_pos_arr = to_numpy(cube.get_pos())
    traj, phases = build_traj(cube_pos_arr, args.gripper_height_above_cube, steps_per_ep)

    ep_data = {"images_up": [], "images_side": [], "cube_z": [], "phase": []}
    z_before = z_after = None

    for fi, (td, phase) in enumerate(zip(traj, phases)):
        tr = np.deg2rad(np.array(td, dtype=np.float32))
        so101.control_dofs_position(tr, dof_idx)
        scene.step()
        z = float(to_numpy(cube.get_pos())[2])
        if phase == "close" and z_before is None:
            z_before = z
        if phase == "lift":
            z_after = z
        ep_data["images_up"].append(render_camera(cam_up))
        ep_data["images_side"].append(render_camera(cam_side))
        ep_data["cube_z"].append(z)
        ep_data["phase"].append(phase)

    if z_before is None:
        z_before = float(to_numpy(cube.get_pos())[2])
    if z_after is None:
        z_after = float(to_numpy(cube.get_pos())[2])
    delta = z_after - z_before
    success = delta > args.lift_threshold
    print(f"  RESULT: {'SUCCESS' if success else 'FAIL'}, delta_z={delta:.4f}m (before={z_before:.4f}, after={z_after:.4f})")

    # Save
    out_dir = Path(args.save) / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    close_phases = {"close", "close_hold"}
    close_idx = [i for i, p in enumerate(ep_data["phase"]) if p in close_phases]
    if close_idx:
        start = max(0, close_idx[0] - 8)
        end = min(len(ep_data["phase"]), close_idx[-1] + 12)
        png_dir = out_dir / "debug_pngs"
        png_dir.mkdir(exist_ok=True)
        for i in range(start, end):
            stitched = np.concatenate([ep_data["images_up"][i], ep_data["images_side"][i]], axis=1)
            save_rgb_png(stitched, png_dir / f"f{i:03d}_{ep_data['phase'][i]}.png")
        print(f"  Debug PNGs: {png_dir}")

    metrics = {
        "exp_id": args.exp_id,
        "ee_link": "gripper",
        "ik_orientation": "quat_down" if use_quat else "free",
        "platform_height": ph,
        "cube_center_z": float(cube_z),
        "gripper_height_above_cube": args.gripper_height_above_cube,
        "gripper_close": args.gripper_close,
        "close_hold_steps": args.close_hold_steps,
        "cube_z_before_close": z_before,
        "cube_z_after_lift": z_after,
        "cube_lift_delta": delta,
        "grasp_success": int(success),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  {args.exp_id}: {'SUCCESS' if success else 'FAIL'}  delta_z={delta:.4f}m")
    print(f"  platform={ph}, h_above={args.gripper_height_above_cube:.3f}, quat={'down' if use_quat else 'free'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
