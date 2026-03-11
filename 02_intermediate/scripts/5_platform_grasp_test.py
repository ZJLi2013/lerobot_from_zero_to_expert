"""
SO-101 platform grasp test — raise cube into the arm's reachable workspace.

Key finding: the SO-101's minimum jaw height is ~z=0.07m. A 3cm cube on a flat
table (center z=0.015) is unreachable.  This script adds a platform box to
raise the cube into the graspable zone, then runs a single pick attempt.

Approach strategy: two-phase lateral approach to avoid knocking the cube during descent.
  1. Descend to grasp height at a lateral offset (beside the cube)
  2. Slide laterally into position (jaws around the cube)
  3. Close, hold, lift

Usage:
  python 5_platform_grasp_test.py --platform-height 0.06
  python 5_platform_grasp_test.py --platform-height 0.06 --sweep-approach-z
  python 5_platform_grasp_test.py --platform-height 0.06 --auto-tune-offset
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
    subprocess.Popen(
        ["Xvfb", ":99", "-screen", "0", "1280x1024x24", "-ac", "+extension", "GLX"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    os.environ["DISPLAY"] = ":99"
    time.sleep(2)


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


JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
HOME_DEG = np.array([0.0, -30.0, 90.0, -60.0, 0.0, 0.0], dtype=np.float32)
KP = np.array([500.0, 500.0, 400.0, 400.0, 300.0, 200.0])
KV = np.array([50.0, 50.0, 40.0, 40.0, 30.0, 20.0])


def main():
    ensure_display()
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=None)
    ap.add_argument("--exp-id", default="P1_platform")
    ap.add_argument("--save", default="/output")
    ap.add_argument("--platform-height", type=float, default=0.06)
    ap.add_argument("--cube-size", type=float, default=0.03)
    ap.add_argument("--cube-x", type=float, default=0.16)
    ap.add_argument("--cube-y", type=float, default=0.0)
    ap.add_argument("--approach-z", type=float, default=-0.065,
                    help="Approach z offset relative to cube center (for grasp_center target)")
    ap.add_argument("--lateral-offset-y", type=float, default=0.06,
                    help="Lateral offset in y for safe descent (avoid hitting cube)")
    ap.add_argument("--gripper-open", type=float, default=0.0)
    ap.add_argument("--gripper-close", type=float, default=-20.0)
    ap.add_argument("--close-hold-steps", type=int, default=50)
    ap.add_argument("--lift-threshold", type=float, default=0.01)
    ap.add_argument("--auto-tune-offset", action="store_true")
    ap.add_argument("--sweep-approach-z", action="store_true")
    ap.add_argument("--offset-x-cands", default="-0.008,-0.004,0.0,0.004,0.008")
    ap.add_argument("--offset-y-cands", default="-0.008,-0.004,0.0,0.004,0.008")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--episode-length", type=float, default=10.0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    # ── Locate MJCF ──
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
    jaw_box_cfg = load_jaw_box_config(xml_path)

    # ── Build scene ──
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
    scene.add_entity(
        gs.morphs.Box(size=(0.10, 0.10, ph), pos=(args.cube_x, args.cube_y, ph / 2), fixed=True),
        surface=gs.surfaces.Default(color=(0.6, 0.6, 0.6, 1.0)),
    )

    cs = args.cube_size
    cube_z = ph + cs / 2
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
    print(f"Scene built: platform_h={ph}, cube_center_z={cube_z:.4f}")

    # ── Setup joints ──
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

    ee_link = None
    ee_name = None
    for c in ["grasp_center", "moving_jaw_so101_v1", "gripper"]:
        try:
            ee_link = so101.get_link(c)
            ee_name = c
            break
        except Exception:
            pass
    print(f"EE link: {ee_name}")

    gripper_link = so101.get_link("gripper")
    try:
        mj_link = so101.get_link("moving_jaw_so101_v1")
    except Exception:
        mj_link = None

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

    # IK sanity
    ik_target = np.array([args.cube_x, args.cube_y, cube_z])
    q_test = so101.inverse_kinematics(
        link=ee_link, pos=ik_target, quat=None,
        init_qpos=home_rad, max_solver_iters=50, damping=0.02,
    )
    q_np = to_numpy(q_test)
    so101.set_qpos(q_np)
    for _ in range(15):
        scene.step()
    ee_pos = to_numpy(ee_link.get_pos())
    ik_err = np.linalg.norm(ee_pos - ik_target)
    print(f"IK sanity: target={ik_target.tolist()}, err={ik_err:.4f}m")

    if mj_link:
        grip_p = to_numpy(gripper_link.get_pos())
        mj_p = to_numpy(mj_link.get_pos())
        fixed_box = get_jaw_box_world(gripper_link, jaw_box_cfg["fixed_jaw_box"])
        moving_box = get_jaw_box_world(mj_link, jaw_box_cfg["moving_jaw_box"])
        fixed_axis = fixed_box["thickness_axis_world"]
        moving_axis = moving_box["thickness_axis_world"]
        fixed_to_moving = moving_box["center_world"] - fixed_box["center_world"]
        moving_to_fixed = fixed_box["center_world"] - moving_box["center_world"]
        fixed_inward = fixed_axis * np.sign(np.dot(fixed_to_moving, fixed_axis) or 1.0)
        moving_inward = moving_axis * np.sign(np.dot(moving_to_fixed, moving_axis) or 1.0)
        fixed_inner = fixed_box["center_world"] + fixed_inward * fixed_box["half_thickness"]
        moving_inner = moving_box["center_world"] + moving_inward * moving_box["half_thickness"]
        jaw_mid = 0.5 * (fixed_inner + moving_inner)
        print(
            f"  jaw_mid=[{jaw_mid[0]:.4f},{jaw_mid[1]:.4f},{jaw_mid[2]:.4f}], "
            f"jaw_mid_z-cube_z={jaw_mid[2]-cube_z:+.4f}"
        )

    so101.set_qpos(home_rad)
    for _ in range(30):
        scene.step()

    # ── Helpers ──
    def solve_ik(pos, grip_deg, seed_rad=None):
        q = to_numpy(so101.inverse_kinematics(
            link=ee_link, pos=pos, quat=None,
            init_qpos=seed_rad, max_solver_iters=50, damping=0.02,
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

    def build_lateral_traj(cube_pos, ox, oy, approach_z_val, total_steps):
        """Two-phase lateral approach: descend to the SIDE of the cube, then slide in."""
        lat_y = args.lateral_offset_y
        target_z = approach_z_val

        # Waypoints (all positions are grasp_center targets)
        pos_high = cube_pos + np.array([ox, oy, 0.10])
        pos_side_high = cube_pos + np.array([ox, lat_y, 0.10])
        pos_side_low = cube_pos + np.array([ox, lat_y, target_z])
        pos_approach = cube_pos + np.array([ox, oy, target_z])
        pos_lift = cube_pos + np.array([ox, oy, 0.12])

        q_high = solve_ik(pos_high, args.gripper_open, seed_rad=home_rad)

        # Phase 1: home -> high above cube
        traj, phases = [], []
        n_move = max(12, total_steps // 10)
        traj += lerp(home_deg.copy(), q_high, n_move)
        phases += ["move_pre"] * n_move

        # Phase 2: move laterally to side position (same height)
        q_side_high = solve_ik(pos_side_high, args.gripper_open,
                               seed_rad=np.deg2rad(np.array(q_high, dtype=np.float32)))
        n_lat1 = max(8, total_steps // 12)
        traj += lerp(q_high, q_side_high, n_lat1)
        phases += ["move_side"] * n_lat1

        # Phase 3: descend at the side (safe — cube is not here)
        n_desc = 8
        prev = q_side_high
        prev_rad = np.deg2rad(np.array(prev, dtype=np.float32))
        for i in range(n_desc):
            frac = (i + 1) / n_desc
            z = 0.10 + (target_z - 0.10) * frac
            p = cube_pos + np.array([ox, lat_y, z])
            wp = solve_ik(p, args.gripper_open, seed_rad=prev_rad)
            spw = max(3, total_steps // (10 * n_desc))
            traj += lerp(prev, wp, spw)
            phases += ["descend_side"] * spw
            prev = wp
            prev_rad = np.deg2rad(np.array(wp, dtype=np.float32))

        q_side_low = prev

        # Phase 4: slide laterally into grasp position
        n_slide = 6
        prev_rad = np.deg2rad(np.array(q_side_low, dtype=np.float32))
        prev = q_side_low
        for i in range(n_slide):
            frac = (i + 1) / n_slide
            y = lat_y + (oy - lat_y) * frac
            p = cube_pos + np.array([ox, y, target_z])
            wp = solve_ik(p, args.gripper_open, seed_rad=prev_rad)
            spw = max(3, total_steps // (10 * n_slide))
            traj += lerp(prev, wp, spw)
            phases += ["slide_in"] * spw
            prev = wp
            prev_rad = np.deg2rad(np.array(wp, dtype=np.float32))

        q_approach = prev

        # Phase 5: close gripper
        q_close = q_approach.copy()
        if n_dofs >= 6:
            q_close[5] = args.gripper_close
        n_close = max(10, total_steps // 12)
        traj += lerp(q_approach, q_close, n_close)
        phases += ["close"] * n_close

        # Phase 6: hold
        n_hold = args.close_hold_steps
        traj += [q_close.copy() for _ in range(n_hold)]
        phases += ["close_hold"] * n_hold

        # Phase 7: lift
        q_close_rad = np.deg2rad(np.array(q_close, dtype=np.float32))
        n_lift_wps = 4
        prev = q_close
        prev_rad = q_close_rad
        for i in range(n_lift_wps):
            frac = (i + 1) / n_lift_wps
            z = target_z + (0.12 - target_z) * frac
            p = cube_pos + np.array([ox, oy, z])
            wp = solve_ik(p, args.gripper_close, seed_rad=prev_rad)
            slw = max(5, total_steps // (10 * n_lift_wps))
            traj += lerp(prev, wp, slw)
            phases += ["lift"] * slw
            prev = wp
            prev_rad = np.deg2rad(np.array(wp, dtype=np.float32))

        # Phase 8: return
        consumed = len(traj)
        n_ret = max(0, total_steps - consumed)
        if n_ret > 0:
            traj += lerp(prev, home_deg.copy(), n_ret)
            phases += ["return"] * n_ret

        return traj, phases

    def run_trial(cube_pos, ox, oy, approach_z_val):
        reset_scene(settle=20)
        traj, phases = build_lateral_traj(cube_pos, ox, oy, approach_z_val, total_steps=120)
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

    cube_pos_arr = np.array([args.cube_x, args.cube_y, cube_z])
    reset_scene()
    cube_pos_arr = to_numpy(cube.get_pos())
    print(f"\ncube settled at [{cube_pos_arr[0]:.4f}, {cube_pos_arr[1]:.4f}, {cube_pos_arr[2]:.4f}]")

    # ── Approach-z sweep ──
    if args.sweep_approach_z:
        print("\n=== Approach-z sweep (lateral approach) ===")
        print(f"{'approach_z':>12} {'delta_z':>10} {'note':>10}")
        print("-" * 40)
        best_az = args.approach_z
        best_dz = -1e9
        for az in np.arange(-0.08, 0.02, 0.005):
            try:
                dz = run_trial(cube_pos_arr, 0.0, 0.0, az)
            except Exception as e:
                print(f"  error at az={az:.3f}: {e}")
                dz = -1.0
            tag = "***" if dz > args.lift_threshold else ""
            print(f"{az:12.3f} {dz:10.4f} {tag:>10}")
            if dz > best_dz:
                best_dz = dz
                best_az = az
        print(f"\nbest approach_z = {best_az:.3f}, delta_z = {best_dz:.4f}")
        args.approach_z = best_az

    # ── Offset auto-tune ──
    chosen_ox, chosen_oy = 0.0, 0.0
    search_log = []
    if args.auto_tune_offset:
        print("\n=== Offset auto-tune (lateral approach) ===")
        xc = [float(x) for x in args.offset_x_cands.split(",")]
        yc = [float(y) for y in args.offset_y_cands.split(",")]
        best_dz = -1e9
        best_xy = (0.0, 0.0)
        total = len(xc) * len(yc)
        tried = 0
        for ox in xc:
            for oy in yc:
                tried += 1
                try:
                    dz = run_trial(cube_pos_arr, ox, oy, args.approach_z)
                except Exception:
                    dz = -1.0
                search_log.append({"ox": ox, "oy": oy, "dz": float(dz)})
                tag = "***" if dz > args.lift_threshold else "   "
                if tried <= 30 or dz > args.lift_threshold:
                    print(f"  {tag} [{tried}/{total}] ox={ox:+.3f} oy={oy:+.3f} dz={dz:+.4f}")
                if dz > best_dz:
                    best_dz = dz
                    best_xy = (ox, oy)
        chosen_ox, chosen_oy = best_xy
        print(f"  best offset=({chosen_ox:+.3f}, {chosen_oy:+.3f}), dz={best_dz:+.4f}")

    # ── Full episode ──
    print(f"\n=== Full episode: approach_z={args.approach_z:.3f}, offset=({chosen_ox:+.3f}, {chosen_oy:+.3f}) ===")
    steps_per_ep = int(args.episode_length * args.fps)
    reset_scene(settle=30)
    cube_pos_arr = to_numpy(cube.get_pos())
    traj, phases = build_lateral_traj(cube_pos_arr, chosen_ox, chosen_oy, args.approach_z, steps_per_ep)

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

    # ── Save ──
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
        print(f"  Debug PNGs: {png_dir} (frames {start}-{end - 1})")

    for label, idx in [("slide_in", max(0, close_idx[0] - 3) if close_idx else 0),
                       ("close_start", close_idx[0] if close_idx else 0),
                       ("close_hold", close_idx[len(close_idx) // 2] if close_idx else 0),
                       ("lift_start", close_idx[-1] if close_idx else 0)]:
        stitched = np.concatenate([ep_data["images_up"][idx], ep_data["images_side"][idx]], axis=1)
        save_rgb_png(stitched, out_dir / f"{label}.png")

    metrics = {
        "exp_id": args.exp_id,
        "ee_link": ee_name,
        "platform_height": ph,
        "cube_center_z": float(cube_z),
        "cube_settled_z": float(cube_pos_arr[2]),
        "approach_z": args.approach_z,
        "lateral_offset_y": args.lateral_offset_y,
        "gripper_close": args.gripper_close,
        "close_hold_steps": args.close_hold_steps,
        "offset": [chosen_ox, chosen_oy],
        "cube_z_before_close": z_before,
        "cube_z_after_lift": z_after,
        "cube_lift_delta": delta,
        "grasp_success": int(success),
        "search_log": search_log,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  metrics.json: {out_dir}")
    print(f"\n{'='*60}")
    print(f"  {args.exp_id}: {'SUCCESS' if success else 'FAIL'}  delta_z={delta:.4f}m")
    print(f"  platform={ph}m  cube_z={cube_z:.4f}  approach_z={args.approach_z:.3f}")
    print(f"  lateral_y={args.lateral_offset_y}  offset=({chosen_ox:+.3f}, {chosen_oy:+.3f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
