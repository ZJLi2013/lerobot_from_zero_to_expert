"""Sweep grasp_center z-offset to find the best compromise between IK accuracy and jaw proximity.

Instead of modifying the XML each time, we use inverse_kinematics with different
target positions to simulate what would happen if grasp_center were at various
offsets from the gripper body.

The key relationship: when IK places grasp_center at target T,
the jaw midpoint ends up at some position J.  We want J ≈ cube center.
"""
import argparse
import numpy as np
import xml.etree.ElementTree as ET
import genesis as gs


def to_np(t):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True)
    parser.add_argument("--cube-z", type=float, default=0.015)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 30, substeps=4),
        rigid_options=gs.options.RigidOptions(enable_collision=True, enable_joint_limit=True),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=(0.15, 0.0, args.cube_z)))
    so101 = scene.add_entity(gs.morphs.MJCF(file=args.xml, pos=(0, 0, 0)))
    scene.build()
    jaw_box_cfg = load_jaw_box_config(args.xml)

    home_rad = np.deg2rad([0, -30, 90, -60, 0, 0]).astype(np.float32)
    dof_idx = np.arange(6)
    so101.set_dofs_kp(np.array([500, 500, 400, 400, 300, 200.0]), dof_idx)
    so101.set_dofs_kv(np.array([50, 50, 40, 40, 30, 20.0]), dof_idx)

    gc = so101.get_link("grasp_center")
    gripper_link = so101.get_link("gripper")
    mj = so101.get_link("moving_jaw_so101_v1")

    def get_jaw_box_world(link, cfg):
        link_pos = to_np(link.get_pos())
        link_quat = to_np(link.get_quat())
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

    cube_center = np.array([0.16, 0.0, args.cube_z])
    cube_top = args.cube_z + 0.015

    print(f"cube center = {cube_center.tolist()}, cube top z = {cube_top:.4f}")
    print(f"{'target_z':>10} {'gc_err':>8} {'gc_z':>8} {'grip_z':>8} {'mj_z':>8} {'jaw_mid_z':>10} {'gap_to_top':>11} {'verdict':>10}")
    print("-" * 95)

    best_score = 1e9
    best_tz = None

    for target_z in np.arange(-0.08, 0.12, 0.005):
        target = np.array([0.16, 0.0, target_z])
        so101.set_qpos(home_rad)
        so101.control_dofs_position(home_rad, dof_idx)
        for _ in range(10):
            scene.step()

        q = so101.inverse_kinematics(
            link=gc, pos=target, quat=None,
            init_qpos=home_rad, max_solver_iters=50, damping=0.02,
        )
        q_np = to_np(q)
        so101.set_qpos(q_np)
        so101.control_dofs_position(q_np, dof_idx)
        for _ in range(20):
            scene.step()

        gc_pos = to_np(gc.get_pos())
        grip_pos = to_np(gripper_link.get_pos())
        mj_pos = to_np(mj.get_pos())
        gc_err = np.linalg.norm(gc_pos - target)
        fixed_box = get_jaw_box_world(gripper_link, jaw_box_cfg["fixed_jaw_box"])
        moving_box = get_jaw_box_world(mj, jaw_box_cfg["moving_jaw_box"])
        fixed_axis = fixed_box["thickness_axis_world"]
        moving_axis = moving_box["thickness_axis_world"]
        fixed_to_moving = moving_box["center_world"] - fixed_box["center_world"]
        moving_to_fixed = fixed_box["center_world"] - moving_box["center_world"]
        fixed_inward = fixed_axis * np.sign(np.dot(fixed_to_moving, fixed_axis) or 1.0)
        moving_inward = moving_axis * np.sign(np.dot(moving_to_fixed, moving_axis) or 1.0)
        fixed_inner = fixed_box["center_world"] + fixed_inward * fixed_box["half_thickness"]
        moving_inner = moving_box["center_world"] + moving_inward * moving_box["half_thickness"]
        jaw_mid_z = float(0.5 * (fixed_inner[2] + moving_inner[2]))
        gap = jaw_mid_z - cube_top

        if gc_err < 0.005:
            verdict = "GOOD_IK"
        elif gc_err < 0.02:
            verdict = "ok_ik"
        else:
            verdict = "bad_ik"

        if gc_err < 0.01 and abs(gap) < 0.02:
            verdict = "*** BEST"
            score = gc_err + abs(gap)
            if score < best_score:
                best_score = score
                best_tz = target_z

        print(f"{target_z:10.3f} {gc_err:8.4f} {gc_pos[2]:8.4f} {grip_pos[2]:8.4f} {mj_pos[2]:8.4f} {jaw_mid_z:10.4f} {gap:11.4f} {verdict:>10}")

    print(f"\nBest target_z = {best_tz}")
    if best_tz is not None:
        offset_needed = best_tz - args.cube_z
        print(f"  This means approach_z + offset_z should sum to {offset_needed:.4f}")
        print(f"  e.g. approach_z={offset_needed:.4f}, offset_z=0")
        print(f"  or approach_z={offset_needed + 0.01:.4f}, offset_z=-0.01")
    else:
        print("  No configuration found with both good IK and jaw near cube.")
        print("  The arm workspace may not reach this cube position.")
        print("  Try raising the cube or moving it closer to the base.")


if __name__ == "__main__":
    main()
