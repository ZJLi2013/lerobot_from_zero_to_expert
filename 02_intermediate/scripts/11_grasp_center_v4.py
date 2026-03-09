"""
V4 grasp_center frame diagnostic and geometry-based reconstruction.

Purpose:
- Visualize the current `grasp_center` frame axes against the actual jaw corridor
- Reconstruct a candidate local `grasp_center.quat` from jaw geometry
- Estimate a geometry-based residual local position correction
- Export a composite PNG and JSON summary for manual review

Notes:
- This script follows the v4 design in `doc/calib.md`: verify axis semantics first,
  then rebuild the TCP frame from geometry, then estimate the residual position error.
- It uses jaw box collision geoms from `so101_new_calib_v3_jawbox.xml` to approximate
  the true pinch corridor more robustly than raw STL contact meshes.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def ensure_display():
    if os.environ.get("DISPLAY"):
        print(f"[display] DISPLAY={os.environ['DISPLAY']}")
        return
    xvfb_path = shutil.which("Xvfb")
    if not xvfb_path:
        print("[display] WARNING: Xvfb not found")
        return
    print("[display] Starting Xvfb :99 ...")
    proc = subprocess.Popen(
        [xvfb_path, ":99", "-screen", "0", "1280x1024x24", "-ac", "+extension", "GLX"],
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


def normalize(vec, fallback=None):
    vec = np.array(vec, dtype=np.float64)
    n = np.linalg.norm(vec)
    if n < 1e-12:
        if fallback is None:
            raise ValueError("Cannot normalize near-zero vector")
        return normalize(fallback)
    return vec / n


def quat_to_rotmat(quat_wxyz):
    w, x, y, z = np.array(quat_wxyz, dtype=np.float64)
    n = np.linalg.norm([w, x, y, z])
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = np.array([w, x, y, z], dtype=np.float64) / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotmat_to_quat(rot):
    r = np.array(rot, dtype=np.float64)
    trace = np.trace(r)
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    if quat[0] < 0:
        quat = -quat
    return quat / np.linalg.norm(quat)


def angle_deg(a, b):
    a = normalize(a)
    b = normalize(b)
    cosv = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.degrees(math.acos(cosv))


def transform_point(link_pos, link_quat, local_pos):
    return np.array(link_pos, dtype=np.float64) + quat_to_rotmat(link_quat) @ np.array(
        local_pos, dtype=np.float64
    )


def parse_vec(text):
    return np.array([float(x) for x in text.split()], dtype=np.float64)


def parse_grasp_center_and_jaw_boxes(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("worldbody not found in XML")

    data = {
        "grasp_center": None,
        "fixed_jaw_box": None,
        "moving_jaw_box": None,
    }

    def walk(body):
        body_name = body.attrib.get("name")
        if body_name == "grasp_center":
            data["grasp_center"] = {
                "body_name": body_name,
                "pos": parse_vec(body.attrib["pos"]),
                "quat": parse_vec(body.attrib["quat"]),
            }
        for geom in body.findall("geom"):
            geom_name = geom.attrib.get("name")
            if geom_name in {"fixed_jaw_box", "moving_jaw_box"}:
                data[geom_name] = {
                    "geom_name": geom_name,
                    "body_name": body_name,
                    "pos": parse_vec(geom.attrib["pos"]),
                    "size": parse_vec(geom.attrib["size"]),
                }
        for child in body.findall("body"):
            walk(child)

    for body in worldbody.findall("body"):
        walk(body)

    missing = [key for key, value in data.items() if value is None]
    if missing:
        raise RuntimeError(
            "Required XML entries missing. This diagnostic expects jawbox XML. "
            f"Missing: {missing}"
        )
    return data


def build_projection_bounds(points, dim_a, dim_b, margin=0.02):
    a_vals = [float(p[dim_a]) for p in points]
    b_vals = [float(p[dim_b]) for p in points]
    a_min, a_max = min(a_vals), max(a_vals)
    b_min, b_max = min(b_vals), max(b_vals)
    if a_max - a_min < 1e-6:
        a_min -= 0.05
        a_max += 0.05
    if b_max - b_min < 1e-6:
        b_min -= 0.05
        b_max += 0.05
    return (a_min - margin, a_max + margin, b_min - margin, b_max + margin)


def project_point(point, bounds, dim_a, dim_b, width, height, pad=24):
    a_min, a_max, b_min, b_max = bounds
    usable_w = width - 2 * pad
    usable_h = height - 2 * pad
    x = pad + usable_w * (float(point[dim_a]) - a_min) / max(a_max - a_min, 1e-9)
    y = pad + usable_h * (b_max - float(point[dim_b])) / max(b_max - b_min, 1e-9)
    return (x, y)


def draw_arrow(draw, p0, p1, color, width=3):
    draw.line([p0, p1], fill=color, width=width)
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return
    ux, uy = dx / norm, dy / norm
    left = (p1[0] - 10 * ux + 4 * uy, p1[1] - 10 * uy - 4 * ux)
    right = (p1[0] - 10 * ux - 4 * uy, p1[1] - 10 * uy + 4 * ux)
    draw.line([left, p1, right], fill=color, width=width)


def draw_projection_panel(draw, origin_xy, size_xy, title, bounds, dim_a, dim_b, diag, axis_len):
    try:
        from PIL import ImageColor
    except ImportError:
        ImageColor = None

    ox, oy = origin_xy
    width, height = size_xy
    draw.rectangle([ox, oy, ox + width, oy + height], outline=(180, 180, 180), width=2)
    draw.text((ox + 10, oy + 8), title, fill=(20, 20, 20))

    def panel_point(point):
        px, py = project_point(point, bounds, dim_a, dim_b, width, height)
        return (ox + px, oy + py)

    fixed_p = panel_point(diag["fixed_inner_surface_world"])
    moving_p = panel_point(diag["moving_inner_surface_world"])
    jaw_mid_p = panel_point(diag["jaw_midpoint_world"])
    cube_p = panel_point(diag["cube_center_world"])
    gc_p = panel_point(diag["current_gc_world"])
    gripper_p = panel_point(diag["gripper_world"])

    draw.line([fixed_p, moving_p], fill=(70, 70, 70), width=3)
    draw.ellipse([fixed_p[0] - 4, fixed_p[1] - 4, fixed_p[0] + 4, fixed_p[1] + 4], fill=(30, 30, 30))
    draw.ellipse(
        [moving_p[0] - 4, moving_p[1] - 4, moving_p[0] + 4, moving_p[1] + 4], fill=(30, 30, 30)
    )
    draw.ellipse([jaw_mid_p[0] - 5, jaw_mid_p[1] - 5, jaw_mid_p[0] + 5, jaw_mid_p[1] + 5], fill=(170, 0, 170))
    draw.rectangle([cube_p[0] - 4, cube_p[1] - 4, cube_p[0] + 4, cube_p[1] + 4], fill=(210, 40, 40))
    draw.ellipse([gc_p[0] - 4, gc_p[1] - 4, gc_p[0] + 4, gc_p[1] + 4], fill=(0, 70, 160))
    draw.ellipse(
        [gripper_p[0] - 4, gripper_p[1] - 4, gripper_p[0] + 4, gripper_p[1] + 4], fill=(100, 100, 100)
    )

    current_colors = {"x": (220, 60, 60), "y": (60, 170, 60), "z": (60, 90, 220)}
    rebuilt_colors = {"x": (255, 140, 0), "y": (160, 60, 220), "z": (0, 170, 170)}
    for axis_name, axis_dir in diag["current_axes_world"].items():
        end_p = panel_point(np.array(diag["current_gc_world"]) + axis_len * np.array(axis_dir))
        draw_arrow(draw, gc_p, end_p, current_colors[axis_name], width=3)
    for axis_name, axis_dir in diag["rebuilt_axes_world"].items():
        end_p = panel_point(np.array(diag["jaw_midpoint_world"]) + axis_len * np.array(axis_dir))
        draw_arrow(draw, jaw_mid_p, end_p, rebuilt_colors[axis_name], width=2)

    label_y = oy + height - 38
    draw.text((ox + 10, label_y), "red/green/blue=current x/y/z", fill=(20, 20, 20))
    draw.text((ox + 10, label_y + 16), "orange/purple/cyan=rebuild x/y/z", fill=(20, 20, 20))


def build_composite_figure(render_img, diag, out_path):
    from PIL import Image, ImageDraw

    render_img = Image.fromarray(render_img)
    panel_w, panel_h = 420, 300
    text_w = 420
    canvas_w = render_img.width + text_w
    canvas_h = max(render_img.height, panel_h * 2)
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(250, 250, 250))
    canvas.paste(render_img, (0, 0))
    draw = ImageDraw.Draw(canvas)

    points = [
        np.array(diag["cube_center_world"]),
        np.array(diag["jaw_midpoint_world"]),
        np.array(diag["fixed_inner_surface_world"]),
        np.array(diag["moving_inner_surface_world"]),
        np.array(diag["current_gc_world"]),
        np.array(diag["gripper_world"]),
        np.array(diag["jaw_midpoint_world"]) + 0.04 * np.array(diag["rebuilt_axes_world"]["x"]),
        np.array(diag["jaw_midpoint_world"]) + 0.04 * np.array(diag["rebuilt_axes_world"]["y"]),
        np.array(diag["jaw_midpoint_world"]) + 0.04 * np.array(diag["rebuilt_axes_world"]["z"]),
        np.array(diag["current_gc_world"]) + 0.04 * np.array(diag["current_axes_world"]["x"]),
        np.array(diag["current_gc_world"]) + 0.04 * np.array(diag["current_axes_world"]["y"]),
        np.array(diag["current_gc_world"]) + 0.04 * np.array(diag["current_axes_world"]["z"]),
    ]
    bounds_xy = build_projection_bounds(points, 0, 1)
    bounds_xz = build_projection_bounds(points, 0, 2)
    bounds_yz = build_projection_bounds(points, 1, 2)

    x0 = render_img.width
    draw_projection_panel(
        draw,
        (x0, 0),
        (panel_w, panel_h),
        "Top View (XY)",
        bounds_xy,
        0,
        1,
        diag,
        axis_len=0.04,
    )
    draw_projection_panel(
        draw,
        (x0, panel_h),
        (panel_w, panel_h),
        "Side View (XZ)",
        bounds_xz,
        0,
        2,
        diag,
        axis_len=0.04,
    )

    tx = render_img.width + 10
    ty = panel_h * 2 + 10 if panel_h * 2 + 10 < canvas_h else panel_h * 2 - 180
    summary_lines = [
        f"exp_id: {diag['exp_id']}",
        f"xml: {Path(diag['xml_path']).name}",
        "",
        "Current local grasp_center:",
        f"  pos = {np.round(diag['current_gc_local_pos'], 6).tolist()}",
        f"  quat = {np.round(diag['current_gc_local_quat'], 6).tolist()}",
        "",
        "Geometry rebuild suggestion:",
        f"  pos = {np.round(diag['suggested_gc_local_pos'], 6).tolist()}",
        f"  quat = {np.round(diag['suggested_gc_local_quat'], 6).tolist()}",
        "",
        "Axis mismatch (deg):",
        f"  x(corridor): {diag['axis_angle_error_deg']['x']:.2f}",
        f"  y(closing):  {diag['axis_angle_error_deg']['y']:.2f}",
        f"  z(approach): {diag['axis_angle_error_deg']['z']:.2f}",
        "",
        "Residual correction:",
        f"  world delta = {np.round(diag['residual_world_delta'], 6).tolist()}",
        f"  local delta = {np.round(diag['residual_local_delta'], 6).tolist()}",
        "",
        "Legend:",
        "  black line = jaw inner surfaces",
        "  magenta = jaw midpoint",
        "  red square = cube center",
        "  blue point = current grasp_center",
    ]
    y = 610
    for line in summary_lines:
        draw.text((tx, y), line, fill=(20, 20, 20))
        y += 16

    canvas.save(out_path)


def parse_csv_floats(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main():
    ensure_display()

    parser = argparse.ArgumentParser(description="SO-101 v4 grasp_center frame diagnostic")
    parser.add_argument(
        "--xml",
        default="02_intermediate/scripts/assets/so101_new_calib_v3_jawbox.xml",
        help="Path to jawbox XML",
    )
    parser.add_argument("--exp-id", default="V4_frame_diag")
    parser.add_argument("--save", default="/output")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--img-w", type=int, default=640)
    parser.add_argument("--img-h", type=int, default=480)
    parser.add_argument("--cube-x", type=float, default=0.16)
    parser.add_argument("--cube-y", type=float, default=0.0)
    parser.add_argument("--cube-z", type=float, default=0.015)
    parser.add_argument("--approach-z", type=float, default=0.012)
    parser.add_argument("--grasp-offset-x", type=float, default=0.0)
    parser.add_argument("--grasp-offset-y", type=float, default=0.0)
    parser.add_argument("--grasp-offset-z", type=float, default=-0.010)
    parser.add_argument("--gripper-open", type=float, default=20.0)
    parser.add_argument("--gripper-close", type=float, default=-10.0)
    parser.add_argument("--close-hold-steps", type=int, default=50)
    parser.add_argument("--cube-friction", type=float, default=None)
    parser.add_argument("--warmup-auto-tune", action="store_true")
    parser.add_argument("--force-offset", action="store_true")
    parser.add_argument("--offset-x-candidates", default="-0.008,-0.004,0.0,0.004,0.008")
    parser.add_argument("--offset-y-candidates", default="-0.008,-0.004,0.0,0.004,0.008")
    parser.add_argument("--offset-z-candidates", default="-0.010")
    parser.add_argument("--trial-steps", type=int, default=90)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    xml_path = Path(args.xml)
    xml_cfg = parse_grasp_center_and_jaw_boxes(xml_path)

    import torch
    import genesis as gs

    JOINT_NAMES = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]
    HOME_DEG = np.array([0.0, -30.0, 90.0, -60.0, 0.0, 0.0], dtype=np.float32)
    KP = np.array([500.0, 500.0, 400.0, 400.0, 300.0, 200.0], dtype=np.float32)
    KV = np.array([50.0, 50.0, 40.0, 40.0, 30.0, 20.0], dtype=np.float32)

    backend = gs.cpu if args.cpu else gs.gpu
    gs.init(backend=backend, logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / args.fps, substeps=4),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            enable_joint_limit=True,
            box_box_detection=True,
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
    cube = scene.add_entity(**cube_kw)
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

    home_rad = np.deg2rad(HOME_DEG)
    dof_idx = [so101.get_joint(n).dof_idx_local for n in JOINT_NAMES]
    so101.set_dofs_kp(KP, dof_idx)
    so101.set_dofs_kv(KV, dof_idx)
    so101.set_qpos(home_rad)
    so101.control_dofs_position(home_rad, dof_idx)
    for _ in range(60):
        scene.step()

    ee_link = so101.get_link("grasp_center")
    gripper_link = so101.get_link("gripper")
    fixed_jaw_link = so101.get_link(xml_cfg["fixed_jaw_box"]["body_name"])
    moving_jaw_link = so101.get_link(xml_cfg["moving_jaw_box"]["body_name"])

    def reset_scene(settle_steps=30):
        so101.set_qpos(home_rad)
        so101.control_dofs_position(home_rad, dof_idx)
        so101.zero_all_dofs_velocity()
        cube_pos = torch.tensor(
            [args.cube_x, args.cube_y, args.cube_z], dtype=torch.float32, device=gs.device
        ).unsqueeze(0)
        cube.set_pos(cube_pos)
        cube.set_quat(torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=gs.device).unsqueeze(0))
        cube.zero_all_dofs_velocity()
        for _ in range(settle_steps):
            scene.step()

    def solve_ik_seeded(pos, grip_deg, seed_rad=None):
        q = to_numpy(
            so101.inverse_kinematics(
                link=ee_link,
                pos=np.array(pos, dtype=np.float32),
                quat=None,
                init_qpos=seed_rad if seed_rad is not None else home_rad,
                max_solver_iters=50,
                damping=0.02,
            )
        )
        q_deg = np.rad2deg(q).astype(np.float64)
        q_deg[5] = grip_deg
        return q_deg

    def lerp(a, b, n):
        a = np.array(a, dtype=np.float64)
        b = np.array(b, dtype=np.float64)
        return [a + (b - a) * (i + 1) / max(n, 1) for i in range(n)]

    def build_trial_trajectory(cube_pos, offset_x, offset_y, offset_z, total_steps):
        off = np.array([offset_x, offset_y, offset_z], dtype=np.float64)
        q_pre = solve_ik_seeded(cube_pos + off + np.array([0.0, 0.0, 0.10]), args.gripper_open, home_rad)
        q_pre_rad = np.deg2rad(np.array(q_pre, dtype=np.float32))

        descent_wps = []
        prev_rad = q_pre_rad
        for i in range(6):
            frac = (i + 1) / 6
            z = 0.10 + (args.approach_z - 0.10) * frac
            pos = cube_pos + off + np.array([0.0, 0.0, z], dtype=np.float64)
            wp = solve_ik_seeded(pos, args.gripper_open, prev_rad)
            descent_wps.append(wp)
            prev_rad = np.deg2rad(np.array(wp, dtype=np.float32))

        q_approach = descent_wps[-1]
        q_close = q_approach.copy()
        q_close[5] = args.gripper_close

        q_close_rad = np.deg2rad(np.array(q_close, dtype=np.float32))
        lift_wps = []
        prev_rad = q_close_rad
        for i in range(4):
            frac = (i + 1) / 4
            z = args.approach_z + (0.15 - args.approach_z) * frac
            pos = cube_pos + off + np.array([0.0, 0.0, z], dtype=np.float64)
            wp = solve_ik_seeded(pos, args.gripper_close, prev_rad)
            lift_wps.append(wp)
            prev_rad = np.deg2rad(np.array(wp, dtype=np.float32))

        q_lift = lift_wps[-1]
        traj = []
        phases = []
        n_move = max(15, total_steps // 8)
        traj += lerp(HOME_DEG.copy(), q_pre, n_move)
        phases += ["move_pre"] * n_move

        steps_per_wp = max(3, total_steps // (8 * 6))
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

        steps_per_lift = max(5, total_steps // (8 * 4))
        prev = q_close
        for wp in lift_wps:
            traj += lerp(prev, wp, steps_per_lift)
            phases += ["lift"] * steps_per_lift
            prev = wp

        consumed = len(traj)
        n_return = max(0, total_steps - consumed)
        if n_return > 0:
            traj += lerp(q_lift, HOME_DEG.copy(), n_return)
            phases += ["return"] * n_return
        return traj, phases

    def run_warmup_trial(cube_pos, offset_x, offset_y, offset_z):
        reset_scene(settle_steps=20)
        traj, phases = build_trial_trajectory(cube_pos, offset_x, offset_y, offset_z, args.trial_steps)
        z_before = None
        z_after = None
        for target_deg, phase in zip(traj, phases):
            target_rad = np.deg2rad(np.array(target_deg, dtype=np.float32))
            so101.control_dofs_position(target_rad, dof_idx)
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
        return float(z_after - z_before)

    def get_jaw_box_world(link, cfg):
        link_pos = to_numpy(link.get_pos())
        link_quat = to_numpy(link.get_quat())
        rot = quat_to_rotmat(link_quat)
        center_world = transform_point(link_pos, link_quat, cfg["pos"])
        axes_world = {
            "x": normalize(rot @ np.array([1.0, 0.0, 0.0])),
            "y": normalize(rot @ np.array([0.0, 1.0, 0.0])),
            "z": normalize(rot @ np.array([0.0, 0.0, 1.0])),
        }
        return {
            "center_world": center_world,
            "axes_world": axes_world,
            "size": np.array(cfg["size"], dtype=np.float64),
        }

    cube_pos = np.array([args.cube_x, args.cube_y, args.cube_z], dtype=np.float64)
    warmup_results = []
    selected_offset = np.array(
        [args.grasp_offset_x, args.grasp_offset_y, args.grasp_offset_z], dtype=np.float64
    )
    if args.warmup_auto_tune:
        x_candidates = parse_csv_floats(args.offset_x_candidates)
        y_candidates = parse_csv_floats(args.offset_y_candidates)
        z_candidates = parse_csv_floats(args.offset_z_candidates)
        for ox in x_candidates:
            for oy in y_candidates:
                for oz in z_candidates:
                    lift_delta = run_warmup_trial(cube_pos, ox, oy, oz)
                    warmup_results.append(
                        {
                            "offset_world": [float(ox), float(oy), float(oz)],
                            "lift_delta": float(lift_delta),
                        }
                    )
        warmup_results.sort(key=lambda item: item["lift_delta"], reverse=True)
        if warmup_results and not args.force_offset:
            selected_offset = np.array(warmup_results[0]["offset_world"], dtype=np.float64)

    reset_scene()
    approach_target = cube_pos + np.array(
        [selected_offset[0], selected_offset[1], selected_offset[2] + args.approach_z],
        dtype=np.float64,
    )
    q_approach_deg = solve_ik_seeded(approach_target, args.gripper_open, seed_rad=home_rad)
    q_approach_rad = np.deg2rad(q_approach_deg.astype(np.float32))
    so101.control_dofs_position(q_approach_rad, dof_idx)
    for _ in range(60):
        scene.step()

    gc_world = to_numpy(ee_link.get_pos())
    gc_world_quat = to_numpy(ee_link.get_quat())
    gc_world_rot = quat_to_rotmat(gc_world_quat)
    current_axes_world = {
        "x": gc_world_rot[:, 0].tolist(),
        "y": gc_world_rot[:, 1].tolist(),
        "z": gc_world_rot[:, 2].tolist(),
    }

    gripper_world = to_numpy(gripper_link.get_pos())
    gripper_world_quat = to_numpy(gripper_link.get_quat())
    gripper_world_rot = quat_to_rotmat(gripper_world_quat)

    fixed_box = get_jaw_box_world(fixed_jaw_link, xml_cfg["fixed_jaw_box"])
    moving_box = get_jaw_box_world(moving_jaw_link, xml_cfg["moving_jaw_box"])
    fixed_to_moving = moving_box["center_world"] - fixed_box["center_world"]
    moving_to_fixed = fixed_box["center_world"] - moving_box["center_world"]

    fixed_thickness_axis = fixed_box["axes_world"]["x"]
    moving_thickness_axis = moving_box["axes_world"]["x"]
    fixed_inward = normalize(
        fixed_thickness_axis * np.sign(np.dot(fixed_to_moving, fixed_thickness_axis) or 1.0)
    )
    moving_inward = normalize(
        moving_thickness_axis * np.sign(np.dot(moving_to_fixed, moving_thickness_axis) or 1.0)
    )
    fixed_inner_surface = fixed_box["center_world"] + fixed_inward * float(np.min(fixed_box["size"]))
    moving_inner_surface = moving_box["center_world"] + moving_inward * float(np.min(moving_box["size"]))
    jaw_midpoint = 0.5 * (fixed_inner_surface + moving_inner_surface)

    corridor_guess = normalize(
        fixed_box["axes_world"]["y"] + moving_box["axes_world"]["y"],
        fallback=fixed_box["axes_world"]["y"],
    )
    closing_guess = normalize(moving_inner_surface - fixed_inner_surface)
    approach_guess = normalize(np.cross(corridor_guess, closing_guess))
    approach_sign_ref = gc_world - gripper_world
    if np.dot(approach_guess, approach_sign_ref) < 0:
        approach_guess = -approach_guess
    closing_orth = normalize(np.cross(approach_guess, corridor_guess))
    if np.dot(closing_orth, closing_guess) < 0:
        closing_orth = -closing_orth
    rebuilt_rot_world = np.column_stack([corridor_guess, closing_orth, approach_guess])
    rebuilt_quat_local = rotmat_to_quat(gripper_world_rot.T @ rebuilt_rot_world)

    residual_world_delta = jaw_midpoint - gc_world
    residual_local_delta = gripper_world_rot.T @ residual_world_delta
    suggested_local_pos = xml_cfg["grasp_center"]["pos"] + residual_local_delta

    axis_angle_error_deg = {
        "x": angle_deg(current_axes_world["x"], rebuilt_rot_world[:, 0]),
        "y": angle_deg(current_axes_world["y"], rebuilt_rot_world[:, 1]),
        "z": angle_deg(current_axes_world["z"], rebuilt_rot_world[:, 2]),
    }

    stitched_render = np.concatenate([render_camera(cam_up), render_camera(cam_side)], axis=1)

    out_dir = Path(args.save) / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    render_png = out_dir / "approach_render.png"
    figure_png = out_dir / "frame_diagnostic.png"
    summary_json = out_dir / "frame_diagnostic.json"
    suggestion_txt = out_dir / "suggested_grasp_center_snippet.xml"

    save_rgb_png(stitched_render, render_png)

    diag = {
        "exp_id": args.exp_id,
        "xml_path": str(xml_path),
        "cube_center_world": cube_pos.tolist(),
        "cube_friction": args.cube_friction,
        "warmup_auto_tune": args.warmup_auto_tune,
        "force_offset": args.force_offset,
        "gripper_open_deg": args.gripper_open,
        "gripper_close_deg": args.gripper_close,
        "close_hold_steps": args.close_hold_steps,
        "approach_target_world": approach_target.tolist(),
        "selected_offset_world": selected_offset.tolist(),
        "warmup_results": warmup_results,
        "current_gc_local_pos": xml_cfg["grasp_center"]["pos"].tolist(),
        "current_gc_local_quat": xml_cfg["grasp_center"]["quat"].tolist(),
        "current_gc_world": gc_world.tolist(),
        "current_gc_world_quat": gc_world_quat.tolist(),
        "gripper_world": gripper_world.tolist(),
        "gripper_world_quat": gripper_world_quat.tolist(),
        "current_axes_world": current_axes_world,
        "rebuilt_axes_world": {
            "x": rebuilt_rot_world[:, 0].tolist(),
            "y": rebuilt_rot_world[:, 1].tolist(),
            "z": rebuilt_rot_world[:, 2].tolist(),
        },
        "fixed_inner_surface_world": fixed_inner_surface.tolist(),
        "moving_inner_surface_world": moving_inner_surface.tolist(),
        "jaw_midpoint_world": jaw_midpoint.tolist(),
        "residual_world_delta": residual_world_delta.tolist(),
        "residual_local_delta": residual_local_delta.tolist(),
        "suggested_gc_local_pos": suggested_local_pos.tolist(),
        "suggested_gc_local_quat": rebuilt_quat_local.tolist(),
        "axis_angle_error_deg": axis_angle_error_deg,
        "jaw_gap": float(np.linalg.norm(moving_inner_surface - fixed_inner_surface)),
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=2)

    with open(suggestion_txt, "w", encoding="utf-8") as f:
        f.write(
            "<body name=\"grasp_center\" "
            f"pos=\"{' '.join(f'{v:.9f}' for v in suggested_local_pos)}\" "
            f"quat=\"{' '.join(f'{v:.9f}' for v in rebuilt_quat_local)}\">\n"
            "  <site group=\"3\" name=\"grasp_center_site\" pos=\"0 0 0\" quat=\"1 0 0 0\"/>\n"
            "</body>\n"
        )

    build_composite_figure(stitched_render, diag, figure_png)

    print("=" * 70)
    print("V4 FRAME DIAGNOSTIC")
    print("=" * 70)
    print(f"XML:                       {xml_path}")
    print(f"cube_friction:             {args.cube_friction}")
    print(f"warmup_auto_tune:          {args.warmup_auto_tune}")
    print(f"force_offset:              {args.force_offset}")
    print(f"Selected offset world:     {np.round(selected_offset, 6).tolist()}")
    print(f"Approach target world:     {np.round(approach_target, 6).tolist()}")
    print(f"Current local pos:         {np.round(xml_cfg['grasp_center']['pos'], 6).tolist()}")
    print(f"Current local quat:        {np.round(xml_cfg['grasp_center']['quat'], 6).tolist()}")
    print(f"Suggested local pos:       {np.round(suggested_local_pos, 6).tolist()}")
    print(f"Suggested local quat:      {np.round(rebuilt_quat_local, 6).tolist()}")
    print(f"Residual world delta:      {np.round(residual_world_delta, 6).tolist()}")
    print(f"Residual local delta:      {np.round(residual_local_delta, 6).tolist()}")
    print(f"Axis angle error (deg):    {json.dumps(axis_angle_error_deg)}")
    print(f"Jaw midpoint world:        {np.round(jaw_midpoint, 6).tolist()}")
    print(f"Current grasp_center world:{np.round(gc_world, 6).tolist()}")
    print(f"Render PNG:                {render_png}")
    print(f"Figure PNG:                {figure_png}")
    print(f"Summary JSON:              {summary_json}")
    print(f"XML snippet:               {suggestion_txt}")
    print("=" * 70)


if __name__ == "__main__":
    main()
