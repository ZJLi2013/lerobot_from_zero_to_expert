"""
Reproduce issue-1858 scene with a rigid-body object (instead of MPM elastic).

Default mode is a scripted demo so it can run on remote GPU nodes without teleop hardware.
Use `--mode teleop` to keep the original SO101 leader workflow.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import genesis as gs
import numpy as np


def ensure_display() -> None:
    if os.environ.get("DISPLAY"):
        return
    xvfb = subprocess.run(["which", "Xvfb"], capture_output=True)
    if xvfb.returncode != 0:
        return
    proc = subprocess.Popen(
        ["Xvfb", ":99", "-screen", "0", "1280x1024x24", "-ac", "+extension", "GLX"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.environ["DISPLAY"] = ":99"
    time.sleep(2)
    if proc.poll() is None:
        print(f"[display] Xvfb started (PID={proc.pid})")


def find_so101_xml(user_xml: str | None) -> str:
    if user_xml:
        return user_xml
    here = Path(__file__).resolve().parent
    candidates = [
        here / "assets" / "so101_new_calib.xml",
        here / "assets" / "so101_new_calib_v4.xml",
        Path("assets/so101_new_calib.xml"),
        Path("02_intermediate/scripts/assets/so101_new_calib.xml"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return "assets/so101_new_calib.xml"


def convert_action_to_genesis(action: dict) -> np.ndarray:
    return np.array(
        [
            np.deg2rad(action["shoulder_pan.pos"]),
            np.deg2rad(action["shoulder_lift.pos"]),
            np.deg2rad(action["elbow_flex.pos"]),
            np.deg2rad(action["wrist_flex.pos"]),
            np.deg2rad(action["wrist_roll.pos"]),
            np.deg2rad(action["gripper.pos"]),
        ],
        dtype=np.float32,
    )


def demo_action(step: int) -> np.ndarray:
    # Simple open-loop motion that approaches cube and toggles gripper.
    pan = 0.0
    lift = -25.0 + 2.0 * np.sin(step * 0.02)
    elbow = 85.0 + 4.0 * np.sin(step * 0.018 + 1.0)
    wrist_flex = -58.0 + 2.0 * np.sin(step * 0.022 + 2.0)
    wrist_roll = 0.0
    grip = 18.0 if (step // 80) % 2 == 0 else 2.0
    q_deg = np.array([pan, lift, elbow, wrist_flex, wrist_roll, grip], dtype=np.float32)
    return np.deg2rad(q_deg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue-1858 rigid-body reproduce")
    parser.add_argument("--mode", choices=["demo", "teleop"], default="demo")
    parser.add_argument("--steps", type=int, default=450, help="Only used in demo mode")
    parser.add_argument("--xml", default=None)
    parser.add_argument("--out", default="./so101_teleop_rigid_body.mp4")
    parser.add_argument("--teleop-port", default="/dev/ttyACM0")
    args = parser.parse_args()

    ensure_display()
    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1e-2, substeps=33),
        rigid_options=gs.options.RigidOptions(enable_collision=True, enable_joint_limit=True, box_box_detection=True),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1, 1, 1),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=30,
            max_FPS=30,
        ),
        show_viewer=False,
    )
    cam = scene.add_camera(res=(1280, 960), pos=(1, 1, 1), lookat=(0.0, 0.0, 0.0), fov=30, GUI=False)
    scene.add_entity(gs.morphs.Plane())
    so101 = scene.add_entity(
        gs.morphs.MJCF(
            file=find_so101_xml(args.xml),
            pos=(0.0, 0.0, 0.0),
            euler=(0, 0, 0),
        )
    )
    scene.add_entity(
        material=gs.materials.Rigid(friction=1.0),
        morph=gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.15, 0.15, 0.02),
            euler=(0, 0, 0),
        ),
        surface=gs.surfaces.Default(color=(1.0, 0.4, 0.4, 1.0), vis_mode="visual"),
    )

    joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    dof_indices = [so101.get_joint(name).dof_idx for name in joints]
    scene.build()

    teleop_device = None
    if args.mode == "teleop":
        from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

        teleop_config = SO101LeaderConfig(port=args.teleop_port, id="sutie_leader", use_degrees=True)
        teleop_device = SO101Leader(teleop_config)
        teleop_device.connect()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cam.start_recording()
    try:
        if args.mode == "teleop":
            while True:
                action = teleop_device.get_action()
                dof_positions = convert_action_to_genesis(action)
                so101.control_dofs_position(dof_positions, dof_indices)
                cam.render()
                scene.step()
        else:
            for i in range(args.steps):
                so101.control_dofs_position(demo_action(i), dof_indices)
                cam.render()
                scene.step()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cam.stop_recording(str(out_path))
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
