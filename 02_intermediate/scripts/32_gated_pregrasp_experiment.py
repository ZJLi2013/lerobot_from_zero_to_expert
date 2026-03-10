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
    proc = subprocess.Popen([
        "Xvfb", ":99", "-screen", "0", "1280x1024x24", "-ac", "+extension", "GLX"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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


def save_rgb_png(arr, path):
    try:
        from PIL import Image
        Image.fromarray(arr).save(path)
    except ImportError:
        import imageio.v2 as imageio
        imageio.imwrite(path, arr)


def find_xml(user_path=None):
    if user_path:
        p = Path(user_path)
        return p if p.exists() else None
    for p in [
        Path("assets/so101_new_calib_v4.xml"),
        Path("02_intermediate/scripts/assets/so101_new_calib_v4.xml"),
    ]:
        if p.exists():
            return p
    return None


def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def box_tilt_deg(cube_quat):
    r = quat_to_rotmat(cube_quat)
    z_axis = r @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cosv = abs(float(np.clip(np.dot(z_axis / (np.linalg.norm(z_axis) + 1e-9), np.array([0.0, 0.0, 1.0])), -1.0, 1.0)))
    return float(np.degrees(np.arccos(cosv)))


def lerp(a, b, n):
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    return [a + (b - a) * (i + 1) / max(n, 1) for i in range(n)]


HOME_DEG = np.array([0.0, -30.0, 90.0, -60.0, 0.0, 0.0], dtype=np.float32)
KP = np.array([500.0, 500.0, 400.0, 400.0, 300.0, 200.0], dtype=np.float32)
KV = np.array([50.0, 50.0, 40.0, 40.0, 30.0, 20.0], dtype=np.float32)


def main():
    ensure_display()
    ap = argparse.ArgumentParser(description="Gated pre-grasp experiment aligned to 12")
    ap.add_argument("--exp-id", default="gated_exp3_aligned")
    ap.add_argument("--xml", default=None)
    ap.add_argument("--save", default="/output")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--sim-substeps", type=int, default=4)
    ap.add_argument("--trial-steps", type=int, default=90)
    ap.add_argument("--approach-hold-steps", type=int, default=1)
    ap.add_argument("--settle-steps", type=int, default=30)
    ap.add_argument("--cube-x", type=float, default=0.16)
    ap.add_argument("--cube-y", type=float, default=0.0)
    ap.add_argument("--cube-z", type=float, default=0.015)
    ap.add_argument("--cube-friction", type=float, default=1.5)
    ap.add_argument("--gripper-open", type=float, default=20.0)
    ap.add_argument("--grasp-offset-x", type=float, default=0.0)
    ap.add_argument("--grasp-offset-y", type=float, default=0.0)
    ap.add_argument("--grasp-offset-z", type=float, default=-0.01)
    ap.add_argument("--approach-z", type=float, default=0.012)
    ap.add_argument("--z-safe-offset", type=float, default=0.10)
    ap.add_argument("--level-max-steps", type=int, default=20)
    ap.add_argument("--level-tol-z", type=float, default=0.002)
    ap.add_argument("--level-step-y", type=float, default=0.001)
    ap.add_argument("--export-last-frames", type=int, default=10)
    args = ap.parse_args()

    xml = find_xml(args.xml)
    if xml is None:
        print("xml not found")
        sys.exit(1)

    import torch
    import genesis as gs

    gs.init(backend=(gs.cpu if args.cpu else gs.gpu), logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / args.fps, substeps=args.sim_substeps),
        rigid_options=gs.options.RigidOptions(enable_collision=True, enable_joint_limit=True, box_box_detection=True),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    cube = scene.add_entity(
        morph=gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=(args.cube_x, args.cube_y, args.cube_z)),
        material=gs.materials.Rigid(friction=args.cube_friction),
        surface=gs.surfaces.Default(color=(1.0, 0.3, 0.3, 1.0)),
    )
    so101 = scene.add_entity(gs.morphs.MJCF(file=str(xml), pos=(0.0, 0.0, 0.0)))
    cam_up = scene.add_camera(res=(640, 480), pos=(0.42, 0.34, 0.26), lookat=(0.15, 0.0, 0.08), fov=38, GUI=False)
    cam_side = scene.add_camera(res=(640, 480), pos=(0.5, -0.4, 0.3), lookat=(0.15, 0.0, 0.1), fov=45, GUI=False)
    scene.build()

    dof_idx = np.arange(so101.n_dofs)
    so101.set_dofs_kp(KP[:so101.n_dofs], dof_idx)
    so101.set_dofs_kv(KV[:so101.n_dofs], dof_idx)
    home_deg = HOME_DEG[:so101.n_dofs]
    home_rad = np.deg2rad(home_deg)
    ee = so101.get_link("grasp_center")
    fixed = so101.get_link("gripper")
    moving = so101.get_link("moving_jaw_so101_v1")
    cube_init = np.array([args.cube_x, args.cube_y, args.cube_z], dtype=np.float64)
    off = np.array([args.grasp_offset_x, args.grasp_offset_y, args.grasp_offset_z], dtype=np.float64)

    def reset():
        so101.set_qpos(home_rad)
        so101.control_dofs_position(home_rad, dof_idx)
        so101.zero_all_dofs_velocity()
        cube.set_pos(torch.tensor(cube_init, dtype=torch.float32, device=gs.device).unsqueeze(0))
        cube.set_quat(torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=gs.device).unsqueeze(0))
        cube.zero_all_dofs_velocity()
        for _ in range(args.settle_steps):
            scene.step()

    def solve(pos, seed):
        q = to_numpy(so101.inverse_kinematics(link=ee, pos=np.array(pos, dtype=np.float32), quat=None, init_qpos=seed, max_solver_iters=50, damping=0.02))
        qd = np.rad2deg(q)
        qd[5] = args.gripper_open
        return qd

    def baseline_traj():
        pos_pre = cube_init + off + np.array([0.0, 0.0, 0.10], dtype=np.float64)
        q_pre = solve(pos_pre, home_rad)
        prev_rad = np.deg2rad(np.array(q_pre, dtype=np.float32))
        wps = []
        for i in range(6):
            frac = (i + 1) / 6
            z = 0.10 + (args.approach_z - 0.10) * frac
            q = solve(cube_init + off + np.array([0.0, 0.0, z], dtype=np.float64), prev_rad)
            wps.append(q)
            prev_rad = np.deg2rad(np.array(q, dtype=np.float32))
        q_app = wps[-1]
        traj, phase = [], []
        m = lerp(home_deg.copy(), q_pre, max(15, args.trial_steps // 8))
        traj += m; phase += ["move_pre"] * len(m)
        spw = max(3, args.trial_steps // (8 * 6))
        prev = q_pre
        for wp in wps:
            seg = lerp(prev, wp, spw)
            traj += seg; phase += ["approach"] * len(seg)
            prev = wp
        traj += [q_app.copy() for _ in range(args.approach_hold_steps)]
        phase += ["approach_hold"] * args.approach_hold_steps
        return traj, phase

    def run(traj, phase, out_dir):
        keep = [i for i, p in enumerate(phase) if p == "approach"]
        keep = set(keep[-args.export_last_frames:])
        imgs = []
        for i, q in enumerate(traj):
            so101.control_dofs_position(np.deg2rad(np.array(q, dtype=np.float32)), dof_idx)
            scene.step()
            if i in keep:
                imgs.append((i, np.concatenate([render_camera(cam_up), render_camera(cam_side)], axis=1)))
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, im in imgs:
            save_rgb_png(im, out_dir / f"f{i:03d}_approach.png")
        shift = to_numpy(cube.get_pos()) - cube_init
        return float(np.linalg.norm(shift)), float(box_tilt_deg(to_numpy(cube.get_quat())))

    reset()
    btraj, bphase = baseline_traj()
    b_contact, b_tilt = run(btraj, bphase, Path(args.save) / args.exp_id / "baseline_pngs")

    reset()
    pos_pre = cube_init + off + np.array([0.0, 0.0, 0.10], dtype=np.float64)
    q_pre = solve(pos_pre, home_rad)
    move = lerp(home_deg.copy(), q_pre, max(15, args.trial_steps // 8))
    traj = list(move)
    phase = ["move_pre"] * len(move)
    for q in move:
        so101.control_dofs_position(np.deg2rad(np.array(q, dtype=np.float32)), dof_idx)
        scene.step()
    prev_rad = np.deg2rad(np.array(q_pre, dtype=np.float32))
    x_t, y_t = float(cube_init[0] + off[0]), float(cube_init[1] + off[1])
    z_safe = cube_init[2] + off[2] + args.z_safe_offset
    leveled, final_dz = False, None
    for _ in range(args.level_max_steps):
        q = solve([x_t, y_t, z_safe], prev_rad)
        prev_rad = np.deg2rad(np.array(q, dtype=np.float32))
        so101.control_dofs_position(np.deg2rad(np.array(q, dtype=np.float32)), dof_idx)
        scene.step()
        traj.append(q); phase.append("pre_grasp_level")
        dz = float(to_numpy(moving.get_pos())[2] - to_numpy(fixed.get_pos())[2])
        final_dz = dz
        if abs(dz) < args.level_tol_z:
            leveled = True
            break
        y_t += -np.sign(dz) * args.level_step_y

    spw = max(3, args.trial_steps // (8 * 6))
    prev = traj[-1]
    for i in range(6):
        frac = (i + 1) / 6
        z = z_safe + (args.approach_z - z_safe) * frac
        q = solve(cube_init + off + np.array([0.0, 0.0, z], dtype=np.float64), prev_rad)
        prev_rad = np.deg2rad(np.array(q, dtype=np.float32))
        seg = lerp(prev, q, spw)
        traj += seg; phase += ["approach"] * len(seg)
        prev = q
    traj += [prev.copy() for _ in range(args.approach_hold_steps)]
    phase += ["approach_hold"] * args.approach_hold_steps
    g_contact, g_tilt = run(traj, phase, Path(args.save) / args.exp_id / "gated_pngs")

    out = Path(args.save) / args.exp_id
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "exp_id": args.exp_id,
        "baseline_contact": b_contact,
        "gated_contact": g_contact,
        "baseline_tilt": b_tilt,
        "gated_tilt": g_tilt,
        "leveled": leveled,
        "final_level_dz": final_dz,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
