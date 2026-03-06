# 5_parallel_lerobot.py
"""
Genesis × SO-101 合成数据批量生成脚本

功能：
- N_ENVS 个并行仿真环境
- pick-place 状态机（Home→PreGrasp→Approach→Grasp→Lift→Place→Return）
- 域随机化（目标物体位置/颜色/质量）
- 双相机（俯视 + 腕部）图像采集
- 输出 LeRobot HuggingFace Dataset 格式

用法：
    python 5_parallel_lerobot.py \
        --n_envs 64 \
        --n_episodes 500 \
        --repo_id your_hf_username/so101-genesis-pickplace \
        --push_to_hub
"""

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import genesis as gs
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# ── 常量 ─────────────────────────────────────────────────────────────────────

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# LeRobot motor 名（角度制，单位：度）
MOTOR_NAMES = [f"{j}.pos" for j in JOINT_NAMES]

# SO-101 HOME 姿态（度）
HOME_DEG = np.array([0.0, -30.0, 90.0, -60.0, 0.0, 0.0])

# 夹爪：0=张开，>0=闭合（根据实际 XML 调整）
GRIPPER_OPEN_DEG   = 0.0
GRIPPER_CLOSED_DEG = 25.0

# 关节索引
ARM_DOF_IDX     = np.arange(5)    # shoulder_pan … wrist_roll
GRIPPER_DOF_IDX = np.array([5])
ALL_DOF_IDX     = np.arange(6)


# ── 配置 ─────────────────────────────────────────────────────────────────────

@dataclass
class SDGConfig:
    # 仿真
    xml_path:     str   = "assets/so101_new_calib.xml"
    n_envs:       int   = 64
    ctrl_dt:      float = 1 / 50       # 50 Hz 控制频率
    substeps:     int   = 10

    # 采集
    n_episodes:   int   = 500
    max_steps_per_episode: int = 400   # 每 episode 最多步数

    # 域随机化
    cube_x_range: tuple = (0.10, 0.25)   # 前后范围 (m)
    cube_y_range: tuple = (-0.10, 0.10)  # 左右范围 (m)
    cube_size_range: tuple = (0.025, 0.040)  # 边长范围 (m)

    # 数据集
    repo_id:      str   = "local/so101-genesis-pickplace"
    fps:          int   = 50
    push_to_hub:  bool  = False

    # 渲染
    img_width:    int   = 640
    img_height:   int   = 480
    use_wrist_cam: bool = True    # 是否渲染腕部相机
    show_viewer:  bool  = False


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def deg2rad_batch(deg: np.ndarray) -> np.ndarray:
    """度 → 弧度，支持 (6,) 或 (N, 6)"""
    return np.deg2rad(deg)

def rad2deg_batch(rad) -> np.ndarray:
    """弧度（torch.Tensor 或 np.ndarray） → 度"""
    if isinstance(rad, torch.Tensor):
        rad = rad.cpu().numpy()
    return np.rad2deg(rad)

def random_color() -> tuple:
    r, g, b = np.random.uniform(0.2, 1.0, 3)
    return (float(r), float(g), float(b), 1.0)

def interpolate_joints(
    start: np.ndarray,
    end:   np.ndarray,
    n_steps: int,
) -> list[np.ndarray]:
    """在 start 和 end 之间线性插值 n_steps 个路径点（含终点）"""
    return [
        start + (end - start) * (i + 1) / n_steps
        for i in range(n_steps)
    ]


# ── 状态机 ───────────────────────────────────────────────────────────────────

class PickPlaceStateMachine:
    """
    简单 pick-place 状态机，为单个 env 生成关节角度序列（度）。
    返回 list of (target_deg, gripper_state)
    """

    def __init__(self, so101, ee_link, cube_pos_3d: np.ndarray):
        self.so101     = so101
        self.ee_link   = ee_link
        self.cube_pos  = cube_pos_3d  # (3,) world frame

    def solve_ik_deg(
        self,
        pos:         np.ndarray,
        quat:        np.ndarray,
        gripper_deg: float,
        env_idx:     int = 0,
    ) -> Optional[np.ndarray]:
        """IK 求解，返回 6-DOF 目标角度（度），失败返回 None"""
        try:
            q = self.so101.inverse_kinematics(
                link=self.ee_link,
                pos=pos,
                quat=quat,
                envs_idx=np.array([env_idx]),
            )  # (1, 6) rad
            q_np = q.cpu().numpy()[0]   # (6,)
            q_np[5] = np.deg2rad(gripper_deg)
            return rad2deg_batch(q_np)  # (6,) deg
        except Exception:
            return None

    def plan(self, env_idx: int = 0) -> list[np.ndarray]:
        """
        返回完整轨迹：每个元素是 (6,) 目标角度（度）。
        失败时返回空列表。
        """
        traj = []
        cube = self.cube_pos
        down_quat = np.array([0.0, 1.0, 0.0, 0.0])  # 末端朝下

        # 1. Home
        home = HOME_DEG.copy()
        traj += interpolate_joints(HOME_DEG, home, n_steps=20)

        # 2. Pre-grasp（方块正上方 10 cm）
        pregrasp_pos = cube + np.array([0.0, 0.0, 0.10])
        q_pre = self.solve_ik_deg(pregrasp_pos, down_quat, GRIPPER_OPEN_DEG, env_idx)
        if q_pre is None:
            return []
        traj += interpolate_joints(home, q_pre, n_steps=40)

        # 3. Approach（靠近方块 2 cm）
        approach_pos = cube + np.array([0.0, 0.0, 0.02])
        q_app = self.solve_ik_deg(approach_pos, down_quat, GRIPPER_OPEN_DEG, env_idx)
        if q_app is None:
            return []
        traj += interpolate_joints(q_pre, q_app, n_steps=30)

        # 4. Grasp（闭合夹爪，保持位置）
        q_grasp = q_app.copy()
        q_grasp[5] = GRIPPER_CLOSED_DEG
        traj += interpolate_joints(q_app, q_grasp, n_steps=20)

        # 5. Lift（抬高 15 cm）
        lift_pos = cube + np.array([0.0, 0.0, 0.15])
        q_lift = self.solve_ik_deg(lift_pos, down_quat, GRIPPER_CLOSED_DEG, env_idx)
        if q_lift is None:
            return []
        traj += interpolate_joints(q_grasp, q_lift, n_steps=40)

        # 6. Place（移动到放置区域：右侧 10 cm 偏移）
        place_pos = cube + np.array([0.10, 0.10, 0.06])
        q_place = self.solve_ik_deg(place_pos, down_quat, GRIPPER_CLOSED_DEG, env_idx)
        if q_place is None:
            return []
        traj += interpolate_joints(q_lift, q_place, n_steps=50)

        # 7. Release（张开夹爪）
        q_release = q_place.copy()
        q_release[5] = GRIPPER_OPEN_DEG
        traj += interpolate_joints(q_place, q_release, n_steps=15)

        # 8. Return to Home
        traj += interpolate_joints(q_release, HOME_DEG, n_steps=40)

        return traj


# ── 主采集循环 ────────────────────────────────────────────────────────────────

def build_scene(cfg: SDGConfig):
    """构建 Genesis 场景，返回 (scene, so101, cubes, cam_top, cam_wrist)"""

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=cfg.ctrl_dt, substeps=cfg.substeps),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            enable_joint_limit=True,
            constraint_solver=gs.constraint_solver.Newton,
        ),
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=list(range(min(4, cfg.n_envs))),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.6, -0.5, 0.5),
            camera_lookat=(0.15, 0.0, 0.1),
            camera_fov=45,
        ),
        show_viewer=cfg.show_viewer,
    )

    # ── 地面 ──────────────────────────────────────────────────────────────
    scene.add_entity(gs.morphs.Plane())

    # ── SO-101 ────────────────────────────────────────────────────────────
    so101 = scene.add_entity(
        gs.morphs.MJCF(
            file=cfg.xml_path,
            pos=(0.0, 0.0, 0.0),
        )
    )

    # ── 目标方块（每个 env 一个，域随机化在 build 后设置） ─────────────────
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.03, 0.03, 0.03),
            pos=(0.15, 0.0, 0.015),
        ),
        surface=gs.surfaces.Default(color=(1.0, 0.3, 0.3, 1.0)),
    )

    # ── 相机：俯视 ────────────────────────────────────────────────────────
    cam_top = scene.add_camera(
        res=(cfg.img_width, cfg.img_height),
        pos=(0.0, 0.0, 0.7),
        lookat=(0.15, 0.0, 0.0),
        fov=55,
        GUI=False,
    )

    # ── 相机：侧视 ────────────────────────────────────────────────────────
    cam_side = scene.add_camera(
        res=(cfg.img_width, cfg.img_height),
        pos=(0.5, -0.4, 0.3),
        lookat=(0.15, 0.0, 0.1),
        fov=45,
        GUI=False,
    )

    # ── 构建并行环境 ──────────────────────────────────────────────────────
    scene.build(n_envs=cfg.n_envs)

    # ── PD 增益（build 之后设置） ──────────────────────────────────────────
    kp = np.array([100.0, 100.0, 80.0, 80.0, 60.0, 40.0])
    kv = np.array([10.0,  10.0,   8.0,  8.0,  6.0,  4.0])
    so101.set_dofs_kp(kp, ALL_DOF_IDX)
    so101.set_dofs_kv(kv, ALL_DOF_IDX)

    return scene, so101, cube, cam_top, cam_side


def randomize_env(
    cfg:    SDGConfig,
    cube:   "gs.Entity",
    n_envs: int,
) -> np.ndarray:
    """
    随机化各 env 中方块的位置，返回 (n_envs, 3) 世界坐标。
    """
    x = np.random.uniform(*cfg.cube_x_range, size=n_envs)
    y = np.random.uniform(*cfg.cube_y_range, size=n_envs)
    z = np.full(n_envs, 0.015)
    pos = np.stack([x, y, z], axis=-1)   # (N, 3)

    cube_pos_tensor = torch.tensor(pos, dtype=torch.float32, device=gs.device)
    cube.set_pos(cube_pos_tensor)

    return pos   # (N, 3) numpy，供 IK 规划用


def collect_episode(
    cfg:      SDGConfig,
    scene:    "gs.Scene",
    so101:    "gs.Entity",
    cube:     "gs.Entity",
    cam_top:  "gs.Camera",
    cam_side: "gs.Camera",
    dataset:  LeRobotDataset,
    env_idx:  int,
    cube_pos: np.ndarray,   # (3,)
    task_str: str,
) -> bool:
    """
    采集单个 episode（env_idx 对应的并行环境）。
    返回是否成功（False 表示 IK 规划失败）。
    """
    ee_link = so101.get_link("gripper_link")   # 根据实际 XML 调整

    # 重置该 env 的机器人到 home 姿态
    home_rad = deg2rad_batch(HOME_DEG)
    so101.set_qpos(
        np.tile(home_rad, (1, 1)),
        envs_idx=np.array([env_idx]),
    )
    for _ in range(30):
        scene.step()

    # 规划轨迹
    sm = PickPlaceStateMachine(so101, ee_link, cube_pos)
    trajectory_deg = sm.plan(env_idx=env_idx)
    if not trajectory_deg:
        return False

    # 执行并记录
    for target_deg in trajectory_deg:
        target_rad = deg2rad_batch(target_deg)

        # 控制（并行环境只控制 env_idx）
        so101.control_dofs_position(
            target_rad[np.newaxis, :],   # (1, 6)
            ALL_DOF_IDX,
            envs_idx=np.array([env_idx]),
        )
        scene.step()

        # 读取实际状态（弧度 → 度）
        cur_rad = so101.get_dofs_position(ALL_DOF_IDX, envs_idx=np.array([env_idx]))
        cur_deg = rad2deg_batch(cur_rad)[0]   # (6,)

        # 渲染图像（批量，取 env_idx 这一行）
        rgb_top,  _, _, _ = cam_top.render(rgb=True, depth=False,
                                           segmentation=False, normal=False)
        rgb_side, _, _, _ = cam_side.render(rgb=True, depth=False,
                                            segmentation=False, normal=False)

        # rgb 形状：(n_envs, H, W, 3) 或 (H, W, 3)
        if rgb_top.ndim == 4:
            img_top  = rgb_top[env_idx].cpu().numpy().astype(np.uint8)
            img_side = rgb_side[env_idx].cpu().numpy().astype(np.uint8)
        else:
            img_top  = rgb_top.cpu().numpy().astype(np.uint8)
            img_side = rgb_side.cpu().numpy().astype(np.uint8)

        # 写入 LeRobot 格式
        dataset.add_frame({
            "observation.state":       cur_deg.astype(np.float32),
            "action":                  target_deg.astype(np.float32),
            "observation.images.top":  img_top,
            "observation.images.side": img_side,
            "task":                    task_str,
        })

    dataset.save_episode(task=task_str)
    return True


def create_lerobot_dataset(cfg: SDGConfig) -> LeRobotDataset:
    """创建 LeRobot 数据集（若已存在则追加）"""
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": MOTOR_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": MOTOR_NAMES,
        },
        "observation.images.top": {
            "dtype": "video",
            "shape": (3, 480, 640),
            "names": ["channel", "height", "width"],
        },
        "observation.images.side": {
            "dtype": "video",
            "shape": (3, 480, 640),
            "names": ["channel", "height", "width"],
        },
        "task": {
            "dtype": "string",
            "shape": (1,),
            "names": None,
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=cfg.repo_id,
        fps=cfg.fps,
        features=features,
        robot_type="so101",
        use_videos=True,
    )
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Genesis SO-101 SDG")
    parser.add_argument("--n_envs",      type=int,   default=64)
    parser.add_argument("--n_episodes",  type=int,   default=200)
    parser.add_argument("--repo_id",     type=str,   default="local/so101-genesis-pickplace")
    parser.add_argument("--xml",         type=str,   default="assets/so101_new_calib.xml")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--vis",         action="store_true")
    args = parser.parse_args()

    cfg = SDGConfig(
        xml_path    = args.xml,
        n_envs      = args.n_envs,
        n_episodes  = args.n_episodes,
        repo_id     = args.repo_id,
        push_to_hub = args.push_to_hub,
        show_viewer = args.vis,
    )

    # ── 初始化 Genesis ────────────────────────────────────────────────────
    gs.init(backend=gs.gpu, logging_level="warning")

    # ── 构建场景 ──────────────────────────────────────────────────────────
    print(f"[SDG] 构建场景：{cfg.n_envs} 个并行环境...")
    scene, so101, cube, cam_top, cam_side = build_scene(cfg)
    print("[SDG] 场景构建完成")

    # ── 创建数据集 ────────────────────────────────────────────────────────
    dataset = create_lerobot_dataset(cfg)

    task_descriptions = [
        "pick up the red cube and place it to the right",
        "grasp the cube and lift it up",
        "pick the block and move it to the target location",
    ]

    # ── 主循环：按批次采集 ────────────────────────────────────────────────
    n_collected   = 0
    n_failed      = 0
    t_start       = time.time()

    while n_collected < cfg.n_episodes:
        # 随机化所有 env 的方块位置
        cube_positions = randomize_env(cfg, cube, cfg.n_envs)  # (N, 3)

        # 让物理稳定
        for _ in range(20):
            scene.step()

        # 轮询每个 env 采集一个 episode
        for env_idx in range(cfg.n_envs):
            if n_collected >= cfg.n_episodes:
                break

            task_str = task_descriptions[n_collected % len(task_descriptions)]
            success = collect_episode(
                cfg       = cfg,
                scene     = scene,
                so101     = so101,
                cube      = cube,
                cam_top   = cam_top,
                cam_side  = cam_side,
                dataset   = dataset,
                env_idx   = env_idx,
                cube_pos  = cube_positions[env_idx],
                task_str  = task_str,
            )

            if success:
                n_collected += 1
            else:
                n_failed += 1

            # 进度日志
            if n_collected % 10 == 0 and n_collected > 0:
                elapsed = time.time() - t_start
                rate = n_collected / elapsed
                eta  = (cfg.n_episodes - n_collected) / max(rate, 1e-6)
                print(
                    f"[SDG] {n_collected}/{cfg.n_episodes} episodes "
                    f"| 失败 {n_failed} "
                    f"| {rate:.1f} ep/s "
                    f"| ETA {eta/60:.1f} min"
                )

    # ── 收尾 ──────────────────────────────────────────────────────────────
    dataset.consolidate(run_compute_stats=True)

    if cfg.push_to_hub:
        print(f"[SDG] 推送到 HuggingFace Hub: {cfg.repo_id}")
        dataset.push_to_hub()

    elapsed = time.time() - t_start
    print(f"\n[SDG] 完成！共采集 {n_collected} episodes，失败 {n_failed}，耗时 {elapsed/60:.1f} min")
    print(f"[SDG] 数据集路径：{dataset.root}")


if __name__ == "__main__":
    main()
