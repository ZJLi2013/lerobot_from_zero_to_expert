"""
POC: Genesis + SO-101 数据管线可行性验证
=========================================
验证目标（按阶段 PASS/FAIL 输出）：
  [1] Genesis import + GPU 初始化
  [2] 场景构建（Plane + Box + Camera）
  [3] 物理仿真步进 + 相机渲染
  [4] SO-101 MJCF 加载 & 链接名探测
  [5] PD 增益设置 + Home 姿态归零
  [6] IK 求解
  [7] 关节控制 + 状态读取（deg/rad 转换）
  [8] 采集 N 帧 → 保存 .npy，打印维度

用法：
  python genesis_quick_start.py               # 默认 GPU，自动查找 SO-101 XML
  python genesis_quick_start.py --xml PATH    # 指定 XML 路径
  python genesis_quick_start.py --cpu         # CPU fallback（无 GPU 时）
  python genesis_quick_start.py --frames 20  # 自定义采集帧数
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


# ── Headless display setup（必须在任何 OpenGL/genesis import 之前）────────────
def ensure_display():
    """
    headless 服务器（无显示器）上 genesis 需要一个 X11 display。
    若 DISPLAY 未设置，自动尝试启动 Xvfb :99。
    若 Xvfb 不可用，打印提示后继续（genesis 可能仍失败，但会给出明确错误）。
    """
    if os.environ.get("DISPLAY"):
        print(f"[display] DISPLAY={os.environ['DISPLAY']} (already set)")
        return

    # 尝试启动 Xvfb
    xvfb = subprocess.run(["which", "Xvfb"], capture_output=True)
    if xvfb.returncode != 0:
        print("[display] WARNING: DISPLAY not set and Xvfb not found.")
        print("          Install with:  apt-get install -y xvfb")
        print(
            "          Or run:        export DISPLAY=:99 && Xvfb :99 -screen 0 1280x1024x24 -ac &"
        )
        return

    print("[display] No DISPLAY detected (headless server). Starting Xvfb :99 ...")
    proc = subprocess.Popen(
        ["Xvfb", ":99", "-screen", "0", "1280x1024x24", "-ac", "+extension", "GLX"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.environ["DISPLAY"] = ":99"
    time.sleep(2)
    if proc.poll() is None:
        print(f"[display] Xvfb started (PID={proc.pid}, DISPLAY=:99)")
    else:
        print(
            "[display] WARNING: Xvfb exited immediately, display may not be available."
        )


ensure_display()

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--xml", default=None, help="SO-101 MJCF 路径（留空自动查找）")
parser.add_argument("--cpu", action="store_true", help="使用 CPU backend（调试用）")
parser.add_argument("--frames", type=int, default=30, help="每个阶段采集帧数")
parser.add_argument("--save", default="poc_output", help="结果输出目录")
args = parser.parse_args()

PASS = "✓ PASS"
FAIL = "✗ FAIL"
SKIP = "- SKIP"

results = {}


def stage(name):
    print(f"\n{'─'*60}")
    print(f"  [{name}]")
    print(f"{'─'*60}")


def ok(label, val=""):
    msg = f"  {PASS}  {label}"
    if val:
        msg += f"  →  {val}"
    print(msg)


def err(label, e):
    print(f"  {FAIL}  {label}")
    print(f"         {type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# [1] Genesis import + init
# ─────────────────────────────────────────────────────────────────────────────
stage("1/8  Genesis import + GPU init")
try:
    import numpy as np
    import torch
    import genesis as gs

    ok("import genesis as gs", f"genesis version unknown (no __version__)")
    ok(f"torch {torch.__version__}, CUDA available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        ok(f"GPU", torch.cuda.get_device_name(0))
    results["1"] = True
except Exception as e:
    err("import genesis", e)
    results["1"] = False
    print("\n[!] genesis-world 未安装，请先运行:\n    pip install genesis-world\n")
    sys.exit(1)

try:
    backend = gs.cpu if args.cpu else gs.gpu
    gs.init(backend=backend, logging_level="warning")
    ok(f"gs.init(backend={'cpu' if args.cpu else 'gpu'})")
    results["1_init"] = True
except Exception as e:
    err("gs.init", e)
    results["1_init"] = False
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# [2] 场景构建
# ─────────────────────────────────────────────────────────────────────────────
stage("2/8  场景构建（Plane + Box + Camera）")
try:
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1 / 50, substeps=4),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            enable_joint_limit=True,
        ),
        show_viewer=False,
    )
    ok("gs.Scene() created")

    scene.add_entity(gs.morphs.Plane())
    ok("Plane added")

    cube = scene.add_entity(
        gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=(0.15, 0.0, 0.015)),
        surface=gs.surfaces.Default(color=(1.0, 0.3, 0.3, 1.0)),
    )
    ok("Box (target object) added")

    cam = scene.add_camera(
        res=(320, 240),
        pos=(0.4, -0.3, 0.4),
        lookat=(0.15, 0.0, 0.0),
        fov=55,
        GUI=False,
    )
    ok("Camera added", "320×240")
    results["2"] = True
except Exception as e:
    err("scene setup", e)
    results["2"] = False
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# [3] 物理仿真 + 相机渲染（先 build 不含机器人，纯物理验证）
# ─────────────────────────────────────────────────────────────────────────────
stage("3/8  物理仿真步进 + 相机渲染")


# 先找 SO-101 XML（后面 build 时一起加载）
def find_so101_xml(user_path=None):
    """在常见位置搜索 SO-101 MJCF，返回 Path 或 None"""
    if user_path:
        p = Path(user_path)
        if p.exists():
            return p
        print(f"  [!] 指定路径不存在: {user_path}")
        return None
    candidates = [
        Path("assets/so101_new_calib.xml"),
        Path("02_intermediate/scripts/assets/so101_new_calib.xml"),
    ]
    # 从 lerobot 包内查找
    try:
        import lerobot

        base = Path(lerobot.__file__).parent
        for sub in [
            "common/robot_devices/robots/assets",
            "common/robot_devices/motors/assets",
            "configs/robot",
            ".",
        ]:
            p = base / sub / "so101_new_calib.xml"
            if p.exists():
                candidates.insert(0, p)
    except ImportError:
        pass
    for p in candidates:
        if p.exists():
            return p
    return None


xml_path = find_so101_xml(args.xml)
if xml_path:
    ok(f"SO-101 XML found", str(xml_path))
else:
    print(f"  [!] SO-101 XML 未找到，阶段 4-7 将使用 Franka Panda 替代验证基础管线")
    print(f"      （Franka XML 内置于 genesis）")

# 重建包含机器人的场景（genesis scene 一旦 build 就不能再加实体）
try:
    del scene
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1 / 50, substeps=4),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            enable_joint_limit=True,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    cube = scene.add_entity(
        gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=(0.15, 0.0, 0.015)),
        surface=gs.surfaces.Default(color=(1.0, 0.3, 0.3, 1.0)),
    )
    cam = scene.add_camera(
        res=(320, 240),
        pos=(0.4, -0.3, 0.4),
        lookat=(0.15, 0.0, 0.0),
        fov=55,
        GUI=False,
    )

    # 加载机器人
    USE_SO101 = xml_path is not None
    if USE_SO101:
        robot = scene.add_entity(
            gs.morphs.MJCF(file=str(xml_path), pos=(0.0, 0.0, 0.0))
        )
        ok(f"SO-101 entity added from MJCF")
    else:
        robot = scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")
        )
        ok(f"Franka Panda entity added (SO-101 fallback)")

    scene.build()
    ok("scene.build() done")
    results["2b"] = True
except Exception as e:
    err("scene.build with robot", e)
    results["2b"] = False
    sys.exit(1)

# 物理步进 + 渲染
try:
    t0 = time.time()
    for _ in range(10):
        scene.step()
    ok(f"10 × scene.step()", f"{(time.time()-t0)*1000:.0f} ms")

    rgb, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, normal=False)
    rgb_np = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.array(rgb)
    # 单环境时 shape 可能是 (H,W,3) 或 (1,H,W,3)
    if rgb_np.ndim == 4:
        rgb_np = rgb_np[0]
    ok(
        f"cam.render() → RGB",
        f"shape={rgb_np.shape}, dtype={rgb_np.dtype}, range=[{rgb_np.min()},{rgb_np.max()}]",
    )
    results["3"] = True
except Exception as e:
    err("physics step / render", e)
    results["3"] = False

# ─────────────────────────────────────────────────────────────────────────────
# [4] 机器人链接 & DOF 探测
# ─────────────────────────────────────────────────────────────────────────────
stage("4/8  机器人链接 & DOF 探测")
try:
    n_dofs = robot.n_dofs
    ok(f"n_dofs = {n_dofs}")

    print("  链接列表:")
    link_names = [l.name for l in robot.links]
    for ln in link_names:
        print(f"    • {ln}")

    # 探测末端执行器链接名
    EE_CANDIDATES = [
        "gripper_link",
        "Fixed_Jaw",
        "end_effector",
        "tool0",
        "hand",
        "panda_hand",  # franka fallback
    ]
    ee_link = None
    for candidate in EE_CANDIDATES:
        try:
            ee_link = robot.get_link(candidate)
            ok(f"end-effector link", candidate)
            break
        except Exception:
            pass
    if ee_link is None:
        print(f"  [!] 未找到已知 EE 链接名，IK 阶段将跳过")
        print(f"      候选列表: {EE_CANDIDATES}")
        print(f"      请从上方链接列表中选择正确名称")

    results["4"] = True
    results["ee_link"] = ee_link
except Exception as e:
    err("link / dof probe", e)
    results["4"] = False
    results["ee_link"] = None

# ─────────────────────────────────────────────────────────────────────────────
# [5] PD 增益 + Home 姿态
# ─────────────────────────────────────────────────────────────────────────────
stage("5/8  PD 增益设置 + Home 姿态")
try:
    n_dofs = robot.n_dofs
    all_dof_idx = np.arange(n_dofs)

    if USE_SO101:
        # SO-101: 6 DOF
        kp = np.array([100.0, 100.0, 80.0, 80.0, 60.0, 40.0])[:n_dofs]
        kv = np.array([10.0, 10.0, 8.0, 8.0, 6.0, 4.0])[:n_dofs]
        HOME_DEG = np.array([0.0, -30.0, 90.0, -60.0, 0.0, 0.0])[:n_dofs]
    else:
        # Franka: 9 DOF (7 arm + 2 finger)
        kp = np.array(
            [4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100], dtype=float
        )[:n_dofs]
        kv = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10], dtype=float)[:n_dofs]
        HOME_DEG = np.zeros(n_dofs)

    robot.set_dofs_kp(kp, all_dof_idx)
    robot.set_dofs_kv(kv, all_dof_idx)
    ok("PD gains set")

    HOME_RAD = np.deg2rad(HOME_DEG)
    robot.set_qpos(HOME_RAD)
    for _ in range(30):
        scene.step()
    ok("Home pose reached (30 steps)")

    cur_rad = robot.get_dofs_position(all_dof_idx)
    cur_np = cur_rad.cpu().numpy() if hasattr(cur_rad, "cpu") else np.array(cur_rad)
    if cur_np.ndim > 1:
        cur_np = cur_np[0]
    cur_deg = np.rad2deg(cur_np)
    ok(f"get_dofs_position()", f"deg={np.round(cur_deg, 1).tolist()}")
    results["5"] = True
    results["all_dof_idx"] = all_dof_idx
    results["HOME_RAD"] = HOME_RAD
    results["n_dofs"] = n_dofs
except Exception as e:
    err("PD / home pose", e)
    results["5"] = False

# ─────────────────────────────────────────────────────────────────────────────
# [6] IK 求解
# ─────────────────────────────────────────────────────────────────────────────
stage("6/8  IK 求解")
ee_link = results.get("ee_link")
qpos_target = None

if ee_link is None:
    print(f"  {SKIP}  EE 链接未找到，跳过 IK")
    results["6"] = False
else:
    try:
        # 方块正上方 8cm
        cube_pos_t = cube.get_pos()
        cube_pos = (
            cube_pos_t.cpu().numpy()
            if hasattr(cube_pos_t, "cpu")
            else np.array(cube_pos_t)
        )
        if cube_pos.ndim > 1:
            cube_pos = cube_pos[0]
        target_pos = cube_pos + np.array([0.0, 0.0, 0.08])
        target_quat = np.array([0.0, 1.0, 0.0, 0.0])  # 末端朝下

        t0 = time.time()
        qpos_target = robot.inverse_kinematics(
            link=ee_link,
            pos=target_pos,
            quat=target_quat,
        )
        dt_ik = (time.time() - t0) * 1000

        qpos_np = (
            qpos_target.cpu().numpy()
            if hasattr(qpos_target, "cpu")
            else np.array(qpos_target)
        )
        if qpos_np.ndim > 1:
            qpos_np = qpos_np[0]

        ok(
            f"IK solved",
            f"{dt_ik:.1f} ms, qpos={np.round(np.rad2deg(qpos_np), 1).tolist()} deg",
        )
        results["6"] = True
        results["qpos_target"] = qpos_np
    except Exception as e:
        err("inverse_kinematics", e)
        results["6"] = False

# ─────────────────────────────────────────────────────────────────────────────
# [7] 关节控制 + 状态读取
# ─────────────────────────────────────────────────────────────────────────────
stage("7/8  关节控制 + 状态读取")
try:
    all_dof_idx = results.get("all_dof_idx", np.arange(robot.n_dofs))
    qpos_np = results.get(
        "qpos_target", results.get("HOME_RAD", np.zeros(robot.n_dofs))
    )

    robot.control_dofs_position(qpos_np, all_dof_idx)
    for _ in range(20):
        scene.step()

    cur_rad = robot.get_dofs_position(all_dof_idx)
    cur_np = cur_rad.cpu().numpy() if hasattr(cur_rad, "cpu") else np.array(cur_rad)
    if cur_np.ndim > 1:
        cur_np = cur_np[0]
    cur_deg = np.rad2deg(cur_np)

    ok("control_dofs_position() executed")
    ok("get_dofs_position()", f"deg={np.round(cur_deg, 1).tolist()}")

    tracking_err = np.abs(cur_deg - np.rad2deg(qpos_np)).mean()
    ok(f"tracking error (mean abs)", f"{tracking_err:.2f} deg")
    results["7"] = True
except Exception as e:
    err("joint control / state read", e)
    results["7"] = False

# ─────────────────────────────────────────────────────────────────────────────
# [8] 采集 N 帧，保存 .npy
# ─────────────────────────────────────────────────────────────────────────────
stage(f"8/8  采集 {args.frames} 帧 → .npy")
try:
    all_dof_idx = results.get("all_dof_idx", np.arange(robot.n_dofs))
    qpos_np = results.get(
        "qpos_target", results.get("HOME_RAD", np.zeros(robot.n_dofs))
    )

    states = []
    actions = []
    images = []

    robot.control_dofs_position(qpos_np, all_dof_idx)

    for i in range(args.frames):
        scene.step()

        cur_rad = robot.get_dofs_position(all_dof_idx)
        cur_np = cur_rad.cpu().numpy() if hasattr(cur_rad, "cpu") else np.array(cur_rad)
        if cur_np.ndim > 1:
            cur_np = cur_np[0]
        state_deg = np.rad2deg(cur_np).astype(np.float32)
        action_deg = np.rad2deg(qpos_np).astype(np.float32)

        rgb, _, _, _ = cam.render(
            rgb=True, depth=False, segmentation=False, normal=False
        )
        rgb_np = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.array(rgb)
        if rgb_np.ndim == 4:
            rgb_np = rgb_np[0]
        img = rgb_np.astype(np.uint8)  # (H, W, 3)

        states.append(state_deg)
        actions.append(action_deg)
        images.append(img)

    states = np.stack(states)  # (T, 6)
    actions = np.stack(actions)  # (T, 6)
    images = np.stack(images)  # (T, H, W, 3)

    out_dir = Path(args.save)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "states.npy", states)
    np.save(out_dir / "actions.npy", actions)
    np.save(out_dir / "images.npy", images)

    ok(
        f"states  saved",
        f"shape={states.shape},  range=[{states.min():.1f}, {states.max():.1f}] deg",
    )
    ok(
        f"actions saved",
        f"shape={actions.shape}, range=[{actions.min():.1f}, {actions.max():.1f}] deg",
    )
    ok(
        f"images  saved",
        f"shape={images.shape},  range=[{images.min()}, {images.max()}]",
    )
    ok(f"files saved to", str(out_dir.resolve()))
    results["8"] = True
except Exception as e:
    err("frame collection / save", e)
    results["8"] = False

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'═'*60}")
print("  SUMMARY")
print(f"{'═'*60}")
stage_labels = {
    "1": "Genesis import + GPU init",
    "2": "Scene 构建",
    "3": "物理步进 + 相机渲染",
    "4": "链接 & DOF 探测",
    "5": "PD 增益 + Home 姿态",
    "6": "IK 求解",
    "7": "关节控制 + 状态读取",
    "8": f"采集 {args.frames} 帧 + 保存",
}
all_pass = True
for k, label in stage_labels.items():
    status = results.get(k, False)
    symbol = PASS if status else (SKIP if status is None else FAIL)
    print(f"  [{k}/8]  {symbol}  {label}")
    if status is False:
        all_pass = False

print()
if all_pass:
    print("  🎉 全部阶段通过！数据管线可行，可以进行大规模采集。")
else:
    print("  ⚠️  部分阶段失败，请查看上方错误信息。")
print(f"{'═'*60}\n")
