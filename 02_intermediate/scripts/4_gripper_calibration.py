"""
SO-101 gripper calibration viewer.

Purpose:
- Sweep the gripper joint through a list of values.
- Save stitched debug PNGs from two side cameras.
- Help determine whether larger joint values mean "more open" or "more closed".
"""
import argparse
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


def find_so101_xml(user_path=None):
    if user_path:
        p = Path(user_path)
        if p.exists():
            return p
        return None

    candidates = [
        Path("assets/so101_new_calib.xml"),
        Path("02_intermediate/scripts/assets/so101_new_calib.xml"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


HOME_DEG = np.array([0.0, -30.0, 90.0, -60.0, 0.0, 0.0], dtype=np.float32)
KP = np.array([500.0, 500.0, 400.0, 400.0, 300.0, 200.0], dtype=np.float32)
KV = np.array([50.0, 50.0, 40.0, 40.0, 30.0, 20.0], dtype=np.float32)


def parse_csv_floats(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main():
    ensure_display()

    parser = argparse.ArgumentParser(description="SO-101 gripper calibration sweep")
    parser.add_argument("--xml", default=None, help="Path to so101_new_calib.xml")
    parser.add_argument("--save", default="/output")
    parser.add_argument("--exp-id", default="gripper_calibration")
    parser.add_argument("--img-w", type=int, default=640)
    parser.add_argument("--img-h", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--gripper-values",
        default="-20,-10,0,10,20,30,40,50",
        help="Comma-separated gripper joint values in degrees",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=20,
        help="Scene steps after each new gripper target",
    )
    args = parser.parse_args()

    xml_path = find_so101_xml(args.xml)
    if xml_path is None:
        print("✗ so101_new_calib.xml not found")
        sys.exit(1)
    print(f"✓ XML = {xml_path}")

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
    scene.add_entity(
        gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=(0.15, 0.0, 0.015)),
        surface=gs.surfaces.Default(color=(1.0, 0.3, 0.3, 1.0)),
    )
    so101 = scene.add_entity(gs.morphs.MJCF(file=str(xml_path), pos=(0.0, 0.0, 0.0)))

    cam_a = scene.add_camera(
        res=(args.img_w, args.img_h),
        pos=(0.42, 0.34, 0.26),
        lookat=(0.15, 0.0, 0.08),
        fov=38,
        GUI=False,
    )
    cam_b = scene.add_camera(
        res=(args.img_w, args.img_h),
        pos=(0.5, -0.4, 0.3),
        lookat=(0.15, 0.0, 0.1),
        fov=45,
        GUI=False,
    )
    scene.build()

    n_dofs = so101.n_dofs
    dof_idx = np.arange(n_dofs)
    so101.set_dofs_kp(KP[:n_dofs], dof_idx)
    so101.set_dofs_kv(KV[:n_dofs], dof_idx)

    home_deg = HOME_DEG[:n_dofs].copy()
    home_rad = np.deg2rad(home_deg)
    so101.set_qpos(home_rad)
    so101.control_dofs_position(home_rad, dof_idx)
    for _ in range(50):
        scene.step()

    out_dir = Path(args.save) / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    gripper_values = parse_csv_floats(args.gripper_values)
    print(f"gripper values = {gripper_values}")

    for i, grip_deg in enumerate(gripper_values):
        q = home_deg.copy()
        if n_dofs < 6:
            print("✗ Expected 6 DOFs including gripper")
            sys.exit(1)
        q[5] = grip_deg
        q_rad = np.deg2rad(q.astype(np.float32))
        so101.control_dofs_position(q_rad, dof_idx)
        for _ in range(args.settle_steps):
            scene.step()

        img_a = render_camera(cam_a)
        img_b = render_camera(cam_b)
        stitched = np.concatenate([img_a, img_b], axis=1)
        png_path = out_dir / f"{i:02d}_gripper_{grip_deg:+.1f}.png"
        save_rgb_png(stitched, png_path)
        print(f"✓ {png_path.name}")


if __name__ == "__main__":
    main()
