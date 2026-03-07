"""
SO-101 grasp experiment runner — MJCF version.

Purpose:
- Debug and tune grasp parameters using the official SO-101 MJCF model.
- Auto-offset search to find best gripper→cube alignment.
- Export npy + rrd + metrics.json for each experiment.

Usage:
  python 3_grasp_experiment.py --exp-id E1 --episodes 1 --episode-length 6
  python 3_grasp_experiment.py --exp-id E2_auto --auto-tune-offset
  python 3_grasp_experiment.py --xml /path/to/so101.xml --exp-id E3
"""

import argparse
import inspect
import json
import os
import subprocess
import sys
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


def parse_csv_floats(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def find_so101_xml(user_path=None):
    """Search for so101_new_calib.xml in common locations, auto-download from HF if needed."""
    if user_path:
        p = Path(user_path)
        if p.exists():
            return p
        return None

    candidates = [
        Path("assets/so101_new_calib.xml"),
        Path("02_intermediate/scripts/assets/so101_new_calib.xml"),
    ]
    try:
        import lerobot

        base = Path(lerobot.__file__).parent
        for sub in [
            "common/robot_devices/robots/assets",
            "common/robot_devices/motors/assets",
            "configs/robot",
        ]:
            p = base / sub / "so101_new_calib.xml"
            if p.exists():
                candidates.insert(0, p)
    except ImportError:
        pass

    for p in candidates:
        if p.exists():
            return p

    try:
        from huggingface_hub import snapshot_download

        print("  … downloading SO101 MJCF from Genesis-Intelligence/assets")
        asset_dir = snapshot_download(
            repo_type="dataset",
            repo_id="Genesis-Intelligence/assets",
            allow_patterns="SO101/*",
            max_workers=1,
        )
        p = Path(asset_dir) / "SO101" / "so101_new_calib.xml"
        if p.exists():
            return p
    except Exception as e:
        print(f"  … HF download failed: {e}")

    return None


# ── Constants (proven working with MJCF from 2_collect.py) ────────────────────

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
HOME_DEG = np.array([0.0, -30.0, 90.0, -60.0, 0.0, 0.0], dtype=np.float32)
IK_QUAT_DOWN = np.array([0.0, 1.0, 0.0, 0.0])
KP = np.array([500.0, 500.0, 400.0, 400.0, 300.0, 200.0])
KV = np.array([50.0, 50.0, 40.0, 40.0, 30.0, 20.0])


def stage(name):
    print(f"\n{'─'*60}\n  [{name}]\n{'─'*60}")


def main():
    ensure_display()

    parser = argparse.ArgumentParser(description="SO-101 grasp experiment (MJCF)")
    parser.add_argument("--exp-id", default="E1_mjcf")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episode-length", type=float, default=8.0)
    parser.add_argument("--img-w", type=int, default=640)
    parser.add_argument("--img-h", type=int, default=480)
    parser.add_argument("--save", default="/output")
    parser.add_argument("--xml", default=None, help="Path to so101_new_calib.xml")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--gripper-open", type=float, default=0.0, help="Gripper open angle in degrees"
    )
    parser.add_argument(
        "--gripper-close",
        type=float,
        default=25.0,
        help="Gripper close angle in degrees",
    )
    parser.add_argument(
        "--pre-close-steps",
        type=int,
        default=0,
        help="Number of hold steps at the approach pose before closing",
    )
    parser.add_argument("--close-hold-steps", type=int, default=12)
    parser.add_argument("--lift-threshold", type=float, default=0.01)
    parser.add_argument(
        "--approach-z",
        type=float,
        default=0.02,
        help="Z offset above cube center for approach (m)",
    )
    parser.add_argument("--grasp-offset-x", type=float, default=0.0)
    parser.add_argument("--grasp-offset-y", type=float, default=0.0)
    parser.add_argument("--grasp-offset-z", type=float, default=0.0)
    parser.add_argument("--auto-tune-offset", action="store_true")
    parser.add_argument("--offset-x-candidates", default="-0.01,-0.005,0.0,0.005,0.01")
    parser.add_argument("--offset-y-candidates", default="-0.01,-0.005,0.0,0.005,0.01")
    parser.add_argument("--offset-z-candidates", default="-0.02,-0.01,0.0,0.01")
    parser.add_argument(
        "--cube-x-min",
        type=float,
        default=0.12,
        help="Minimum sampled cube center x in world frame (m)",
    )
    parser.add_argument(
        "--cube-x-max",
        type=float,
        default=0.20,
        help="Maximum sampled cube center x in world frame (m)",
    )
    parser.add_argument(
        "--cube-y-min",
        type=float,
        default=-0.05,
        help="Minimum sampled cube center y in world frame (m)",
    )
    parser.add_argument(
        "--cube-y-max",
        type=float,
        default=0.05,
        help="Maximum sampled cube center y in world frame (m)",
    )
    parser.add_argument(
        "--cube-fixed-x",
        type=float,
        default=None,
        help="If set, fix cube center x to this world-frame value (m)",
    )
    parser.add_argument(
        "--cube-fixed-y",
        type=float,
        default=None,
        help="If set, fix cube center y to this world-frame value (m)",
    )
    parser.add_argument(
        "--cube-fixed-z",
        type=float,
        default=0.015,
        help="Cube center z to use when a fixed cube pose is requested (m)",
    )
    parser.add_argument(
        "--export-close-debug-pngs",
        action="store_true",
        help="Export dense stitched PNGs around the close phase for debugging",
    )
    parser.add_argument(
        "--debug-close-context",
        type=int,
        default=4,
        help="Frames to include before/after the close-related phases in debug PNG export",
    )
    parser.add_argument(
        "--png-only",
        action="store_true",
        help="Only export close debug PNGs; skip npy, rrd, and metrics.json outputs",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=30,
        help="Physics steps to settle after reset_scene before episode",
    )
    parser.add_argument(
        "--sim-dt",
        type=float,
        default=None,
        help="Override sim dt (default: 1/fps). Use 0.002 for stable grasping.",
    )
    parser.add_argument(
        "--cube-friction",
        type=float,
        default=None,
        help="Cube surface friction (default: Genesis default ~0.5). Try 1.0-2.0 for grasping.",
    )
    parser.add_argument("--sim-substeps", type=int, default=4)
    parser.add_argument("--solver-iterations", type=int, default=50)
    parser.add_argument("--solver-tolerance", type=float, default=1e-6)
    parser.add_argument("--solver-ls-iterations", type=int, default=50)
    parser.add_argument("--solver-ls-tolerance", type=float, default=1e-2)
    parser.add_argument("--noslip-iterations", type=int, default=0)
    parser.add_argument("--constraint-timeconst", type=float, default=0.01)
    parser.add_argument(
        "--integrator",
        choices=["approximate_implicitfast", "implicitfast", "Euler"],
        default="approximate_implicitfast",
    )
    parser.add_argument("--use-gjk-collision", action="store_true")
    args = parser.parse_args()

    # ── [1] Locate MJCF ──────────────────────────────────────────────────────
    stage("1/6  定位 SO-101 MJCF")
    xml_path = find_so101_xml(args.xml)
    if xml_path is None:
        print("  ✗ so101_new_calib.xml not found")
        print("    pip install huggingface_hub   OR   --xml /path/to/so101.xml")
        sys.exit(1)
    print(f"  ✓ {xml_path}")

    # ── [2] Genesis + Scene ───────────────────────────────────────────────────
    stage("2/6  Genesis + Scene")
    import torch
    import genesis as gs
    import genesis.utils.geom as gu

    backend = gs.cpu if args.cpu else gs.gpu
    gs.init(backend=backend, logging_level="warning")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    integrator_map = {
        "approximate_implicitfast": gs.integrator.approximate_implicitfast,
        "implicitfast": gs.integrator.implicitfast,
        "Euler": gs.integrator.Euler,
    }

    rigid_kw = dict(enable_collision=True, enable_joint_limit=True, box_box_detection=True)
    if args.integrator != "approximate_implicitfast":
        rigid_kw["integrator"] = integrator_map[args.integrator]
    if args.solver_iterations != 50:
        rigid_kw["iterations"] = args.solver_iterations
    if args.solver_tolerance != 1e-6:
        rigid_kw["tolerance"] = args.solver_tolerance
    if args.solver_ls_iterations != 50:
        rigid_kw["ls_iterations"] = args.solver_ls_iterations
    if args.solver_ls_tolerance != 1e-2:
        rigid_kw["ls_tolerance"] = args.solver_ls_tolerance
    if args.noslip_iterations != 0:
        rigid_kw["noslip_iterations"] = args.noslip_iterations
    if args.constraint_timeconst != 0.01:
        rigid_kw["constraint_timeconst"] = args.constraint_timeconst
    if args.use_gjk_collision:
        rigid_kw["use_gjk_collision"] = True

    sim_dt = args.sim_dt if args.sim_dt is not None else 1.0 / args.fps
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=sim_dt, substeps=args.sim_substeps),
        rigid_options=gs.options.RigidOptions(**rigid_kw),
        show_viewer=False,
    )
    print(f"  sim dt={sim_dt:.4f}, substeps={args.sim_substeps}")
    scene.add_entity(gs.morphs.Plane())

    cube_entity_kw = dict(
        morph=gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=(0.15, 0.0, 0.015)),
        surface=gs.surfaces.Default(color=(1.0, 0.3, 0.3, 1.0)),
    )
    if args.cube_friction is not None:
        cube_entity_kw["material"] = gs.materials.Rigid(friction=args.cube_friction)
        print(f"  cube friction={args.cube_friction} (gs.materials.Rigid)")
    cube = scene.add_entity(**cube_entity_kw)
    so101 = scene.add_entity(gs.morphs.MJCF(file=str(xml_path), pos=(0.0, 0.0, 0.0)))
    print(f"  ✓ SO-101 loaded via MJCF")

    # For grasp debugging, keep one proven side view and reuse the old "up"
    # stream as a second oblique side view from the opposite side. This makes it
    # much easier to judge whether the cube is actually entering the pinch region.
    cam_up = scene.add_camera(
        res=(args.img_w, args.img_h),
        pos=(0.42, 0.34, 0.26),
        lookat=(0.15, 0.0, 0.08),
        fov=38,
        GUI=False,
    )
    cam_side = scene.add_camera(
        res=(args.img_w, args.img_h),
        pos=(0.5, -0.4, 0.3),
        lookat=(0.15, 0.0, 0.1),
        fov=45,
        GUI=False,
    )
    scene.build()
    print("  ✓ scene built")
    print(
        "  camera[up/debug_side2] = pos=(0.42, 0.34, 0.26), lookat=(0.15, 0.0, 0.08), fov=38"
    )
    print(
        "  camera[side]           = pos=(0.5, -0.4, 0.3), lookat=(0.15, 0.0, 0.1), fov=45"
    )
    print(
        f"  cube sampling range    = x[{args.cube_x_min:.3f}, {args.cube_x_max:.3f}], "
        f"y[{args.cube_y_min:.3f}, {args.cube_y_max:.3f}]"
    )
    if args.cube_fixed_x is not None and args.cube_fixed_y is not None:
        print(
            f"  cube fixed pose        = [{args.cube_fixed_x:.4f}, {args.cube_fixed_y:.4f}, {args.cube_fixed_z:.4f}]"
        )

    # ── [3] Joints + EE + PD + Home ──────────────────────────────────────────
    stage("3/6  Joints + PD + Home")
    n_dofs = so101.n_dofs
    dof_idx = np.arange(n_dofs)
    print(f"  n_dofs = {n_dofs}")

    for j in so101.joints:
        print(f"    joint: {j.name}")

    print("  链接列表:")
    for link in so101.links:
        print(f"    • {link.name}")

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
            print(f"  ✓ EE link = {candidate}")
            break
        except Exception:
            pass
    if ee_link is None:
        print("  ✗ EE link not found")
        sys.exit(1)

    kp = KP[:n_dofs]
    kv = KV[:n_dofs]
    so101.set_dofs_kp(kp, dof_idx)
    so101.set_dofs_kv(kv, dof_idx)

    home_deg = HOME_DEG[:n_dofs]
    home_rad = np.deg2rad(home_deg)
    so101.set_qpos(home_rad)
    so101.control_dofs_position(home_rad, dof_idx)
    for _ in range(60):
        scene.step()

    cur_deg = np.rad2deg(to_numpy(so101.get_dofs_position(dof_idx)))
    home_err = np.abs(cur_deg - home_deg).mean()
    print(f"  HOME = {home_deg.tolist()}")
    print(f"  tracking: mean_err={home_err:.2f}°")

    ik_supports_local_point = (
        "local_point" in inspect.signature(so101.inverse_kinematics).parameters
    )
    print(f"  IK supports local_point = {ik_supports_local_point}")

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
                tcp_local_point = gu.inv_transform_by_quat(
                    tcp_world - jaw_pos, jaw_quat
                )
                print(
                    "  TCP proxy (jaw<->gripper midpoint): "
                    f"world=[{tcp_world[0]:.4f}, {tcp_world[1]:.4f}, {tcp_world[2]:.4f}], "
                    f"local=[{tcp_local_point[0]:.4f}, {tcp_local_point[1]:.4f}, {tcp_local_point[2]:.4f}]"
                )
            else:
                print(
                    "  Dual-link proxy (jaw/gripper midpoint): "
                    f"world=[{tcp_world[0]:.4f}, {tcp_world[1]:.4f}, {tcp_world[2]:.4f}], "
                    f"half_span=[{tcp_half_span_world[0]:.4f}, {tcp_half_span_world[1]:.4f}, {tcp_half_span_world[2]:.4f}]"
                )
        except Exception as e:
            print(f"  TCP proxy unavailable: {e}")
    elif ee_name == "grasp_center":
        print("  grasp_center link detected: using fixed TCP from MJCF asset")

    # IK sanity check — verify the solution actually reaches the target
    try:
        ik_target = np.array([0.15, 0.0, 0.08])
        if ik_supports_local_point and tcp_local_point is not None:
            q_test = so101.inverse_kinematics(
                link=ee_link,
                pos=ik_target,
                quat=None,
                local_point=tcp_local_point,
                init_qpos=home_rad,
                max_solver_iters=50,
                damping=0.02,
            )
        elif gripper_ref is not None and tcp_half_span_world is not None:
            q_test = so101.inverse_kinematics_multilink(
                links=[ee_link, gripper_ref],
                poss=[ik_target - tcp_half_span_world, ik_target + tcp_half_span_world],
                quats=None,
                init_qpos=home_rad,
                max_solver_iters=50,
                damping=0.02,
            )
        else:
            q_test = so101.inverse_kinematics(
                link=ee_link,
                pos=ik_target,
                quat=None,
                init_qpos=home_rad,
                max_solver_iters=50,
                damping=0.02,
            )
        q_test_np = to_numpy(q_test) if hasattr(q_test, "cpu") else np.array(q_test)
        so101.set_qpos(q_test_np)
        so101.control_dofs_position(q_test_np, dof_idx)
        for _ in range(15):
            scene.step()
        ee_pos_check = to_numpy(ee_link.get_pos())
        ee_quat_check = to_numpy(ee_link.get_quat())
        grip_pos_check = (
            to_numpy(gripper_ref.get_pos()) if gripper_ref is not None else None
        )
        if ik_supports_local_point and tcp_local_point is not None:
            tcp_check = ee_pos_check + gu.transform_by_quat(
                tcp_local_point, ee_quat_check
            )
        elif grip_pos_check is not None:
            tcp_check = 0.5 * (ee_pos_check + grip_pos_check)
        else:
            tcp_check = ee_pos_check
        ik_err = np.linalg.norm(tcp_check - ik_target)
        print(
            f"  IK sanity: target={ik_target.tolist()}, "
            f"tcp_actual=[{tcp_check[0]:.4f}, {tcp_check[1]:.4f}, {tcp_check[2]:.4f}], "
            f"err={ik_err:.4f}m"
        )
        if ik_err > 0.03:
            print(f"  ✗ IK error too large ({ik_err:.4f}m > 0.03m) — wrong EE link?")
            print(f"    Trying other links for better IK...")
            best_link_name = ee_name
            best_link_err = ik_err
            for link in so101.links:
                try:
                    q_alt = so101.inverse_kinematics(
                        link=so101.get_link(link.name),
                        pos=ik_target,
                        quat=None,
                        init_qpos=home_rad,
                    )
                    q_alt_np = (
                        to_numpy(q_alt) if hasattr(q_alt, "cpu") else np.array(q_alt)
                    )
                    so101.set_qpos(q_alt_np)
                    so101.control_dofs_position(q_alt_np, dof_idx)
                    for _ in range(15):
                        scene.step()
                    alt_pos = to_numpy(so101.get_link(link.name).get_pos())
                    alt_err = np.linalg.norm(alt_pos - ik_target)
                    if alt_err < best_link_err:
                        print(f"    ★ link '{link.name}' better: err={alt_err:.4f}m")
                        best_link_name = link.name
                        best_link_err = alt_err
                except Exception:
                    pass
            if best_link_name != ee_name:
                ee_link = so101.get_link(best_link_name)
                ee_name = best_link_name
                ik_err = best_link_err
                print(f"    → switched to EE link = {ee_name}")
        if ik_err < 0.03:
            print(
                f"  ✓ IK sanity check passed (EE link = {ee_name}, err={ik_err:.4f}m)"
            )
        else:
            print(
                f"  ⚠ IK accuracy limited (err={ik_err:.4f}m), proceeding with best: {ee_name}"
            )
        so101.set_qpos(home_rad)
        for _ in range(30):
            scene.step()
    except Exception as e:
        print(f"  ✗ IK failed: {e}")
        sys.exit(1)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def solve_ik_seeded(pos, grip_deg, seed_rad=None):
        """Solve IK with explicit seed to ensure consistent solutions."""
        if ik_supports_local_point and tcp_local_point is not None:
            q = to_numpy(
                so101.inverse_kinematics(
                    link=ee_link,
                    pos=pos,
                    quat=None,
                    local_point=tcp_local_point,
                    init_qpos=seed_rad,
                    max_solver_iters=50,
                    damping=0.02,
                )
            )
        elif gripper_ref is not None and tcp_half_span_world is not None:
            q = to_numpy(
                so101.inverse_kinematics_multilink(
                    links=[ee_link, gripper_ref],
                    poss=[pos - tcp_half_span_world, pos + tcp_half_span_world],
                    quats=None,
                    init_qpos=seed_rad,
                    max_solver_iters=50,
                    damping=0.02,
                )
            )
        else:
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

    def reset_scene(cube_pos, settle_steps=30):
        """Full reset: position + velocity for robot and cube."""
        so101.set_qpos(home_rad)
        so101.control_dofs_position(home_rad, dof_idx)
        so101.zero_all_dofs_velocity()
        cube.set_pos(
            torch.tensor(cube_pos, dtype=torch.float32, device=gs.device).unsqueeze(0)
        )
        cube.set_quat(
            torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=gs.device).unsqueeze(
                0
            )
        )
        cube.zero_all_dofs_velocity()
        for _ in range(settle_steps):
            scene.step()

    def build_trajectory_chained_ik(
        cube_pos, offset_x, offset_y, offset_z, total_steps, verbose=False
    ):
        """Build trajectory with chained IK seeding: each solve uses the previous solution as seed."""
        off = np.array([offset_x, offset_y, offset_z])

        pos_pre = cube_pos + off + np.array([0.0, 0.0, 0.10])
        pos_approach = cube_pos + off + np.array([0.0, 0.0, args.approach_z])
        pos_lift = cube_pos + off + np.array([0.0, 0.0, 0.15])

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

        if verbose:
            q_approach_rad = np.deg2rad(np.array(q_approach, dtype=np.float32))
            so101.set_qpos(q_approach_rad)
            so101.control_dofs_position(q_approach_rad, dof_idx)
            for _ in range(10):
                scene.step()
            ee_pos = to_numpy(ee_link.get_pos())
            ee_quat = to_numpy(ee_link.get_quat())
            grip_pos = (
                to_numpy(gripper_ref.get_pos()) if gripper_ref is not None else None
            )
            if ik_supports_local_point and tcp_local_point is not None:
                tcp_pos = ee_pos + gu.transform_by_quat(tcp_local_point, ee_quat)
            elif grip_pos is not None:
                tcp_pos = 0.5 * (ee_pos + grip_pos)
            else:
                tcp_pos = ee_pos
            print(
                f"    [diag] IK approach target: [{pos_approach[0]:.4f}, {pos_approach[1]:.4f}, {pos_approach[2]:.4f}]"
            )
            print(
                f"    [diag] EE actual position:  [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]"
            )
            print(
                f"    [diag] TCP actual position: [{tcp_pos[0]:.4f}, {tcp_pos[1]:.4f}, {tcp_pos[2]:.4f}]"
            )
            print(
                f"    [diag] TCP offset from target: [{tcp_pos[0]-pos_approach[0]:.4f}, {tcp_pos[1]-pos_approach[1]:.4f}, {tcp_pos[2]-pos_approach[2]:.4f}]"
            )
            for other_link in so101.links:
                other_pos = to_numpy(so101.get_link(other_link.name).get_pos())
                print(
                    f"    [diag] link '{other_link.name}': [{other_pos[0]:.4f}, {other_pos[1]:.4f}, {other_pos[2]:.4f}]"
                )

        traj = []
        phases = []

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
        if args.pre_close_steps > 0:
            traj += [q_approach.copy() for _ in range(args.pre_close_steps)]
            phases += ["pre_close_hold"] * args.pre_close_steps
        traj += lerp(q_approach, q_close, n_close)
        phases += ["close"] * n_close

        n_hold = args.close_hold_steps
        traj += [q_close.copy() for _ in range(n_hold)]
        phases += ["close_hold"] * n_hold

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

    def run_trial(cube_pos, offset_x, offset_y, offset_z):
        """Quick trial: reset → chained-IK descent → close → lift → measure delta_z."""
        reset_scene(cube_pos, settle_steps=20)

        traj, phases = build_trajectory_chained_ik(
            cube_pos, offset_x, offset_y, offset_z, total_steps=90
        )

        z_before = None
        z_after = None
        for target_deg, phase in zip(traj, phases):
            target_rad_t = np.deg2rad(np.array(target_deg, dtype=np.float32))
            so101.control_dofs_position(target_rad_t, dof_idx)
            scene.step()
            z_now = float(to_numpy(cube.get_pos())[2])
            if phase == "close" and z_before is None:
                z_before = z_now
            if phase == "lift":
                z_after = z_now

        if z_before is None:
            z_before = float(to_numpy(cube.get_pos())[2])
        if z_after is None:
            z_after = float(to_numpy(cube.get_pos())[2])
        return z_after - z_before

    # ── [4] Auto-tune offset (optional) ──────────────────────────────────────
    stage("4/6  Offset 调参")
    steps_per_episode = int(args.episode_length * args.fps)

    metrics = {
        "exp_id": args.exp_id,
        "model": "MJCF",
        "xml_path": str(xml_path),
        "ee_link": ee_name,
        "ik_quat": IK_QUAT_DOWN.tolist(),
        "home_deg": home_deg.tolist(),
        "gripper_open_deg": args.gripper_open,
        "gripper_close_deg": args.gripper_close,
        "approach_z": args.approach_z,
        "close_hold_steps": args.close_hold_steps,
        "lift_threshold_m": args.lift_threshold,
        "episodes": [],
    }

    out_dir = Path(args.save) / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    all_episodes = []

    def export_close_debug_pngs(ep_data, ep_idx):
        close_like = {"pre_close_hold", "close", "close_hold"}
        close_indices = [
            i for i, phase in enumerate(ep_data["phase"]) if phase in close_like
        ]
        if not close_indices:
            return

        start = max(0, close_indices[0] - args.debug_close_context)
        end = min(
            len(ep_data["phase"]), close_indices[-1] + args.debug_close_context + 1
        )
        debug_dir = out_dir / "close_debug_pngs"
        debug_dir.mkdir(parents=True, exist_ok=True)

        for i in range(start, end):
            stitched = np.concatenate(
                [
                    ep_data["observation.images.up"][i],
                    ep_data["observation.images.side"][i],
                ],
                axis=1,
            )
            phase = ep_data["phase"][i]
            png_path = debug_dir / f"ep{ep_idx:02d}_f{i:03d}_{phase}.png"
            save_rgb_png(stitched, png_path)
        print(f"  ✓ exported close debug PNGs: {debug_dir} (frames {start}-{end - 1})")

    for ep in range(args.episodes):
        # Reset
        so101.set_qpos(home_rad)
        so101.control_dofs_position(home_rad, dof_idx)
        for _ in range(50):
            scene.step()

        # Randomize cube unless a fixed debug pose is requested.
        if args.cube_fixed_x is not None and args.cube_fixed_y is not None:
            cube_pos = np.array(
                [
                    args.cube_fixed_x,
                    args.cube_fixed_y,
                    args.cube_fixed_z,
                ]
            )
        else:
            cube_pos = np.array(
                [
                    np.random.uniform(args.cube_x_min, args.cube_x_max),
                    np.random.uniform(args.cube_y_min, args.cube_y_max),
                    0.015,
                ]
            )
        cube.set_pos(
            torch.tensor(cube_pos, dtype=torch.float32, device=gs.device).unsqueeze(0)
        )
        for _ in range(15):
            scene.step()
        cube_pos = to_numpy(cube.get_pos())
        print(f"  cube pos = [{cube_pos[0]:.4f}, {cube_pos[1]:.4f}, {cube_pos[2]:.4f}]")

        # Optional auto-tune offset (grid search over x/y/z)
        chosen_offset = np.array(
            [args.grasp_offset_x, args.grasp_offset_y, args.grasp_offset_z],
            dtype=np.float64,
        )
        if args.auto_tune_offset:
            x_cands = parse_csv_floats(args.offset_x_candidates)
            y_cands = parse_csv_floats(args.offset_y_candidates)
            z_cands = parse_csv_floats(args.offset_z_candidates)
            best_delta = -1e9
            best_xyz = (chosen_offset[0], chosen_offset[1], chosen_offset[2])
            search_log = []
            total = len(x_cands) * len(y_cands) * len(z_cands)
            tried = 0
            for oz in z_cands:
                for ox in x_cands:
                    for oy in y_cands:
                        tried += 1
                        try:
                            delta = run_trial(cube_pos, ox, oy, oz)
                        except Exception:
                            delta = -1.0
                        search_log.append(
                            {"ox": ox, "oy": oy, "oz": oz, "delta_z": float(delta)}
                        )
                        if delta > best_delta:
                            best_delta = delta
                            best_xyz = (ox, oy, oz)
                        tag = "★" if delta > args.lift_threshold else " "
                        if tried <= 20 or delta > args.lift_threshold:
                            print(
                                f"    {tag} [{tried}/{total}] "
                                f"ox={ox:+.3f} oy={oy:+.3f} oz={oz:+.3f} → Δz={delta:+.4f}m"
                            )
            chosen_offset[:] = best_xyz
            print(
                f"  ✓ best offset=({best_xyz[0]:+.3f}, {best_xyz[1]:+.3f}, {best_xyz[2]:+.3f}) Δz={best_delta:+.4f}m"
            )
            metrics["auto_tune"] = {
                "enabled": True,
                "best_offset": [float(v) for v in best_xyz],
                "best_lift_delta": float(best_delta),
                "search_log": search_log,
            }
        else:
            print(
                f"  offset = ({chosen_offset[0]:+.3f}, {chosen_offset[1]:+.3f}, {chosen_offset[2]:+.3f})"
            )
            metrics["auto_tune"] = {"enabled": False}

        metrics["selected_grasp_offset"] = [float(v) for v in chosen_offset]

        # ── [5] Full episode collection with chosen offset ────────────────────
        stage(f"5/6  数据采集 ep {ep}")

        reset_scene(cube_pos, settle_steps=args.settle_steps)

        trajectory, labels = build_trajectory_chained_ik(
            cube_pos,
            chosen_offset[0],
            chosen_offset[1],
            chosen_offset[2],
            total_steps=steps_per_episode,
            verbose=True,
        )

        # The verbose diagnostic above moves the robot/cube — reset before the actual episode
        reset_scene(cube_pos, settle_steps=args.settle_steps)

        ep_data = {
            "observation.state": [],
            "action": [],
            "observation.images.up": [],
            "observation.images.side": [],
            "timestamp": [],
            "frame_index": [],
            "episode_index": [],
            "cube_z": [],
            "phase": [],
        }
        cube_z_before_close = None
        cube_z_after_lift = None

        for fi, (target_deg, phase) in enumerate(zip(trajectory, labels)):
            target_rad_f = np.deg2rad(np.array(target_deg, dtype=np.float32))
            so101.control_dofs_position(target_rad_f, dof_idx)
            scene.step()

            state_deg = np.rad2deg(to_numpy(so101.get_dofs_position(dof_idx)))
            cube_z = float(to_numpy(cube.get_pos())[2])
            if phase == "close" and cube_z_before_close is None:
                cube_z_before_close = cube_z
            if phase == "lift":
                cube_z_after_lift = cube_z

            ep_data["observation.state"].append(state_deg.astype(np.float32))
            ep_data["action"].append(np.array(target_deg, dtype=np.float32))
            ep_data["observation.images.up"].append(render_camera(cam_up))
            ep_data["observation.images.side"].append(render_camera(cam_side))
            ep_data["timestamp"].append(fi / args.fps)
            ep_data["frame_index"].append(fi)
            ep_data["episode_index"].append(ep)
            ep_data["cube_z"].append(cube_z)
            ep_data["phase"].append(phase)

        all_episodes.append(ep_data)
        if args.export_close_debug_pngs:
            export_close_debug_pngs(ep_data, ep)
        if cube_z_before_close is None:
            cube_z_before_close = float(to_numpy(cube.get_pos())[2])
        if cube_z_after_lift is None:
            cube_z_after_lift = float(to_numpy(cube.get_pos())[2])
        lift_delta = cube_z_after_lift - cube_z_before_close
        grasp_success = 1 if lift_delta > args.lift_threshold else 0
        metrics["episodes"].append(
            {
                "episode": ep,
                "cube_pos": cube_pos.tolist(),
                "cube_z_before_close": cube_z_before_close,
                "cube_z_after_lift": cube_z_after_lift,
                "cube_lift_delta": lift_delta,
                "grasp_success": grasp_success,
            }
        )
        print(
            f"  [ep {ep}] grasp={'✓' if grasp_success else '✗'} "
            f"delta_z={lift_delta:.4f}m "
            f"(before={cube_z_before_close:.4f}, after={cube_z_after_lift:.4f})"
        )

    # ── [6] Save ──────────────────────────────────────────────────────────────
    if args.png_only:
        stage("6/6  保存（PNG only）")
        print("  skipping npy / rrd / metrics.json")
        print(f"\n{'═'*60}")
        print(f"  SUMMARY — {args.exp_id}")
        print(f"{'═'*60}")
        total_success = sum(1 for e in metrics["episodes"] if e["grasp_success"])
        total_eps = len(metrics["episodes"])
        print(f"  Grasp success: {total_success}/{total_eps}")
        if metrics.get("auto_tune", {}).get("enabled"):
            best = metrics["auto_tune"]
            print(
                f"  Best offset: {best['best_offset']}, Δz={best['best_lift_delta']:.4f}m"
            )
        print(f"  PNG output: {out_dir / 'close_debug_pngs'}")
        print(f"{'═'*60}\n")
        return

    stage("6/6  保存")
    import rerun as rr

    all_states = np.concatenate(
        [np.stack(e["observation.state"]) for e in all_episodes]
    )
    all_actions = np.concatenate([np.stack(e["action"]) for e in all_episodes])
    all_imgs_up = np.concatenate(
        [np.stack(e["observation.images.up"]) for e in all_episodes]
    )
    all_imgs_side = np.concatenate(
        [np.stack(e["observation.images.side"]) for e in all_episodes]
    )
    all_ts = np.concatenate([np.array(e["timestamp"]) for e in all_episodes])
    all_fi = np.concatenate([np.array(e["frame_index"]) for e in all_episodes])
    all_ei = np.concatenate([np.array(e["episode_index"]) for e in all_episodes])
    all_cube_z = np.concatenate(
        [np.array(e["cube_z"], dtype=np.float32) for e in all_episodes]
    )

    for name, arr in [
        ("states", all_states),
        ("actions", all_actions),
        ("images_up", all_imgs_up),
        ("images_side", all_imgs_side),
        ("timestamps", all_ts),
        ("frame_indices", all_fi),
        ("episode_indices", all_ei),
        ("cube_z", all_cube_z),
    ]:
        np.save(out_dir / f"{name}.npy", arr)
        print(f"  {name}: {arr.shape}")

    rrd_path = out_dir / f"grasp_{args.exp_id}.rrd"
    rr.init(f"so101_grasp_{args.exp_id}", spawn=False)
    n_joints = min(all_states.shape[1], len(JOINT_NAMES))
    for i in range(len(all_states)):
        rr.set_time("frame_index", sequence=int(all_fi[i]))
        rr.set_time("timestamp", timestamp=float(all_ts[i]))
        rr.log("observation.images.up", rr.Image(all_imgs_up[i]))
        rr.log("observation.images.side", rr.Image(all_imgs_side[i]))
        rr.log("object/cube_z", rr.Scalars(float(all_cube_z[i])))
        for j in range(n_joints):
            jn = JOINT_NAMES[j]
            rr.log(f"state/{jn}", rr.Scalars(float(all_states[i, j])))
            rr.log(f"action/{jn}", rr.Scalars(float(all_actions[i, j])))
    rr.save(str(rrd_path))
    rrd_mb = rrd_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ {rrd_path} ({rrd_mb:.1f} MB)")

    metrics["state_range_deg"] = [float(all_states.min()), float(all_states.max())]
    metrics["action_range_deg"] = [float(all_actions.min()), float(all_actions.max())]
    metrics["output_dir"] = str(out_dir)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  ✓ metrics.json")

    # Summary
    total_success = sum(1 for e in metrics["episodes"] if e["grasp_success"])
    total_eps = len(metrics["episodes"])
    print(f"\n{'═'*60}")
    print(f"  SUMMARY — {args.exp_id}")
    print(f"{'═'*60}")
    print(f"  Grasp success: {total_success}/{total_eps}")
    if metrics.get("auto_tune", {}).get("enabled"):
        best = metrics["auto_tune"]
        print(
            f"  Best offset: {best['best_offset']}, Δz={best['best_lift_delta']:.4f}m"
        )
    print(f"  Output: {out_dir}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
