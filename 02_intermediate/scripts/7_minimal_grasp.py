"""
Minimal SO-101 grasp test: run the EXACT trial code path with image recording.

This eliminates the trial/episode divergence by using identical code for both.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np


def ensure_display():
    if os.environ.get("DISPLAY"):
        return
    subprocess.Popen(["Xvfb", ":99", "-screen", "0", "1280x1024x24", "-ac"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.environ["DISPLAY"] = ":99"
    time.sleep(2)


def to_np(t):
    a = t.cpu().numpy() if hasattr(t, "cpu") else np.array(t)
    return a[0] if a.ndim > 1 else a


def save_png(arr, path):
    try:
        from PIL import Image
        Image.fromarray(arr).save(path)
    except ImportError:
        pass


HOME_DEG = np.array([0, -30, 90, -60, 0, 0], dtype=np.float32)
KP = np.array([500, 500, 400, 400, 300, 200.0])
KV = np.array([50, 50, 40, 40, 30, 20.0])


def main():
    ensure_display()
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=None)
    ap.add_argument("--exp-id", default="M1")
    ap.add_argument("--save", default="/output")
    ap.add_argument("--approach-z", type=float, default=-0.065)
    ap.add_argument("--gripper-close", type=float, default=-20.0)
    ap.add_argument("--hold-steps", type=int, default=20)
    ap.add_argument("--offset-x", type=float, default=-0.005)
    ap.add_argument("--offset-y", type=float, default=0.005)
    ap.add_argument("--repeats", type=int, default=5, help="Run multiple trials to check consistency")
    args = ap.parse_args()

    xml_path = args.xml
    if xml_path is None:
        p = Path("/workspace/lfzte/02_intermediate/scripts/assets/so101_new_calib.xml")
        xml_path = str(p) if p.exists() else None
    if xml_path is None:
        from huggingface_hub import snapshot_download
        d = snapshot_download(repo_type="dataset", repo_id="Genesis-Intelligence/assets",
                             allow_patterns="SO101/*", max_workers=1)
        xml_path = str(Path(d) / "SO101" / "so101_new_calib.xml")
    print(f"XML: {xml_path}")

    import torch, genesis as gs
    gs.init(backend=gs.gpu, logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0/30, substeps=4),
        rigid_options=gs.options.RigidOptions(enable_collision=True, enable_joint_limit=True),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    cube = scene.add_entity(
        gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=(0.16, 0.0, 0.015)),
        surface=gs.surfaces.Default(color=(1.0, 0.3, 0.3, 1.0)),
    )
    so101 = scene.add_entity(gs.morphs.MJCF(file=xml_path, pos=(0, 0, 0)))
    cam = scene.add_camera(res=(640, 480),
        pos=(0.5, -0.4, 0.3), lookat=(0.16, 0.0, 0.1), fov=45, GUI=False)
    scene.build()

    n = so101.n_dofs
    di = np.arange(n)
    so101.set_dofs_kp(KP[:n], di)
    so101.set_dofs_kv(KV[:n], di)
    hr = np.deg2rad(HOME_DEG[:n])
    hd = HOME_DEG[:n]

    ee = so101.get_link("grasp_center")
    print(f"EE: grasp_center, n_dofs={n}")

    def ik(pos, grip_deg, seed=None):
        q = to_np(so101.inverse_kinematics(
            link=ee, pos=pos, quat=None, init_qpos=seed, max_solver_iters=50, damping=0.02))
        qd = np.rad2deg(q)
        if n >= 6:
            qd[5] = grip_deg
        return qd

    def lerp(a, b, steps):
        return [a + (b-a)*(i+1)/max(steps, 1) for i in range(steps)]

    def reset():
        so101.set_qpos(hr)
        so101.control_dofs_position(hr, di)
        so101.zero_all_dofs_velocity()
        cube.set_pos(torch.tensor([0.16, 0.0, 0.015], dtype=torch.float32, device=gs.device).unsqueeze(0))
        cube.set_quat(torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=gs.device).unsqueeze(0))
        cube.zero_all_dofs_velocity()
        for _ in range(20):
            scene.step()

    def build_traj(cp, ox, oy):
        off = np.array([ox, oy, 0.0])
        az = args.approach_z
        go = 0.0
        gc = args.gripper_close
        total = 90

        p_pre = cp + off + np.array([0, 0, 0.10])
        q_pre = ik(p_pre, go, seed=hr)
        pr = np.deg2rad(np.array(q_pre, dtype=np.float32))

        # descent waypoints
        desc = []
        for i in range(6):
            z = 0.10 + (az - 0.10) * (i+1)/6
            wp = ik(cp + off + np.array([0, 0, z]), go, seed=pr)
            desc.append(wp)
            pr = np.deg2rad(np.array(wp, dtype=np.float32))

        q_app = desc[-1]
        q_close = q_app.copy()
        if n >= 6:
            q_close[5] = gc

        # lift waypoints
        q_close_r = np.deg2rad(np.array(q_close, dtype=np.float32))
        lifts = []
        pr2 = q_close_r
        for i in range(4):
            z = az + (0.15-az)*(i+1)/4
            wp = ik(cp + off + np.array([0, 0, z]), gc, seed=pr2)
            lifts.append(wp)
            pr2 = np.deg2rad(np.array(wp, dtype=np.float32))

        traj, phases = [], []
        # move_pre
        nm = max(10, total//8)
        traj += lerp(hd.copy(), q_pre, nm); phases += ["pre"]*nm
        # descend
        prev = q_pre
        for wp in desc:
            s = max(2, total//(8*6))
            traj += lerp(prev, wp, s); phases += ["desc"]*s
            prev = wp
        # close
        nc = max(6, total//12)
        traj += lerp(q_app, q_close, nc); phases += ["close"]*nc
        # hold
        traj += [q_close.copy()]*args.hold_steps; phases += ["hold"]*args.hold_steps
        # lift
        prev = q_close
        for wp in lifts:
            s = max(3, total//(8*4))
            traj += lerp(prev, wp, s); phases += ["lift"]*s
            prev = wp
        return traj, phases

    cp = np.array([0.16, 0.0, 0.015])

    # Run multiple trials to check consistency
    print(f"\n=== Running {args.repeats} trials with offset=({args.offset_x:+.3f}, {args.offset_y:+.3f}), az={args.approach_z} ===")
    results = []
    for r in range(args.repeats):
        reset()
        cp_actual = to_np(cube.get_pos())
        traj, phases = build_traj(cp_actual, args.offset_x, args.offset_y)
        zb = za = None
        for td, ph in zip(traj, phases):
            so101.control_dofs_position(np.deg2rad(np.array(td, dtype=np.float32)), di)
            scene.step()
            z = float(to_np(cube.get_pos())[2])
            if ph == "close" and zb is None:
                zb = z
            if ph == "lift":
                za = z
        if zb is None: zb = float(to_np(cube.get_pos())[2])
        if za is None: za = float(to_np(cube.get_pos())[2])
        dz = za - zb
        tag = "SUCCESS" if dz > 0.01 else "fail"
        results.append(dz)
        print(f"  trial {r}: dz={dz:+.4f}  before={zb:.4f} after={za:.4f}  {tag}")

    print(f"\n  mean dz: {np.mean(results):+.4f}")
    print(f"  success rate: {sum(1 for d in results if d > 0.01)}/{len(results)}")

    # Now run one more time with image recording
    print("\n=== Full recording run ===")
    reset()
    cp_actual = to_np(cube.get_pos())
    traj, phases = build_traj(cp_actual, args.offset_x, args.offset_y)
    images, czs, phs = [], [], []
    zb = za = None
    for fi, (td, ph) in enumerate(zip(traj, phases)):
        so101.control_dofs_position(np.deg2rad(np.array(td, dtype=np.float32)), di)
        scene.step()
        z = float(to_np(cube.get_pos())[2])
        if ph == "close" and zb is None:
            zb = z
        if ph == "lift":
            za = z
        rgb, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, normal=False)
        arr = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.array(rgb)
        if arr.ndim == 4: arr = arr[0]
        images.append(arr.astype(np.uint8))
        czs.append(z)
        phs.append(ph)

    if zb is None: zb = float(to_np(cube.get_pos())[2])
    if za is None: za = float(to_np(cube.get_pos())[2])
    dz = za - zb
    success = dz > 0.01
    print(f"  RESULT: {'SUCCESS' if success else 'FAIL'} dz={dz:+.4f} before={zb:.4f} after={za:.4f}")

    # Save
    od = Path(args.save) / args.exp_id
    od.mkdir(parents=True, exist_ok=True)
    # Save key frames
    ci = [i for i, p in enumerate(phs) if p in ("close", "hold")]
    for label, idx in [("pre_close", max(0, ci[0]-3) if ci else 0),
                       ("close", ci[0] if ci else 0),
                       ("hold", ci[len(ci)//2] if ci else 0),
                       ("lift", ci[-1]+2 if ci else 0)]:
        idx = min(idx, len(images)-1)
        save_png(images[idx], od / f"{label}.png")

    json.dump({
        "exp_id": args.exp_id, "approach_z": args.approach_z,
        "offset": [args.offset_x, args.offset_y],
        "gripper_close": args.gripper_close,
        "delta_z": dz, "success": int(success),
        "trial_results": results, "mean_dz": float(np.mean(results)),
        "success_rate": f"{sum(1 for d in results if d > 0.01)}/{len(results)}",
    }, open(od / "metrics.json", "w"), indent=2)
    print(f"  saved to {od}")


if __name__ == "__main__":
    main()
