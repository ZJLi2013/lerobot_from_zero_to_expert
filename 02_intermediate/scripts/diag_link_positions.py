"""Quick diagnostic: print world positions of all SO-101 links under different IK targets."""
import numpy as np
import genesis as gs

gs.init(backend=gs.gpu, logging_level="warning")
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=1.0 / 30, substeps=4),
    rigid_options=gs.options.RigidOptions(enable_collision=True, enable_joint_limit=True),
    show_viewer=False,
)
scene.add_entity(gs.morphs.Plane())
cube = scene.add_entity(gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=(0.15, 0.0, 0.015)))
so101 = scene.add_entity(
    gs.morphs.MJCF(
        file="/workspace/lfzte/02_intermediate/scripts/assets/so101_new_calib.xml",
        pos=(0, 0, 0),
    )
)
scene.build()

home_rad = np.deg2rad([0, -30, 90, -60, 0, 0]).astype(np.float32)
dof_idx = np.arange(6)
so101.set_dofs_kp(np.array([500, 500, 400, 400, 300, 200.0]), dof_idx)
so101.set_dofs_kv(np.array([50, 50, 40, 40, 30, 20.0]), dof_idx)
so101.set_qpos(home_rad)
so101.control_dofs_position(home_rad, dof_idx)
for _ in range(60):
    scene.step()


def to_np(t):
    arr = t.cpu().numpy() if hasattr(t, "cpu") else np.array(t)
    return arr[0] if arr.ndim > 1 else arr


def print_links(label):
    print(f"\n=== {label} ===")
    for link in so101.links:
        pos = to_np(link.get_pos())
        print(f"  {link.name:30s} world=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")


def ik_to(link_name, target, label):
    link = so101.get_link(link_name)
    q = so101.inverse_kinematics(
        link=link, pos=target, quat=None,
        init_qpos=home_rad, max_solver_iters=50, damping=0.02,
    )
    q_np = to_np(q)
    so101.set_qpos(q_np)
    so101.control_dofs_position(q_np, dof_idx)
    for _ in range(30):
        scene.step()
    actual = to_np(link.get_pos())
    err = np.linalg.norm(actual - target)
    print(f"\nIK target: {link_name} -> {target.tolist()}, actual=[{actual[0]:.4f}, {actual[1]:.4f}, {actual[2]:.4f}], err={err:.4f}m")
    print_links(label)


print_links("HOME configuration")

target = np.array([0.16, 0.0, 0.015])
ik_to("grasp_center", target, "grasp_center -> cube center")
ik_to("moving_jaw_so101_v1", target, "moving_jaw -> cube center")
ik_to("gripper", target, "gripper -> cube center")

target_above = np.array([0.16, 0.0, 0.05])
ik_to("gripper", target_above, "gripper -> 5cm above cube")

target_mid = np.array([0.16, 0.0, 0.03])
ik_to("gripper", target_mid, "gripper -> cube top")
