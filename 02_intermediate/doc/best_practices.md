# SO-101 Genesis SDG Best Practices

> 面向可复用的合成数据流程：稳定朝向、可达抓取、可解释调参、可量化验收。

EE = End Effector（末端执行器），TCP = Tool Center Point（工具中心点），Jaw pinch center = 夹持中心点。
完整术语表和概念关系图见 [grasp_explain.md](grasp_explain.md) 第 1 节。



---

## 1) 推荐基线配置

### 1.1 模型与场景

- **MJCF（推荐）**：`so101_new_calib.xml`，来自 HuggingFace `Genesis-Intelligence/assets`
  - 脚本自动通过 `huggingface_hub.snapshot_download` 下载
  - 也可手动指定 `--xml /path/to/so101_new_calib.xml`
- Genesis 加载：`gs.morphs.MJCF(file=..., pos=(0,0,0))`，无需 `fixed=True` 或 `base_offset`

### 1.2 末端执行器

- 当前 SO-101 MJCF 实际可见链接：
  - `base`, `shoulder`, `upper_arm`, `lower_arm`, `wrist`, `gripper`, `grasp_center`, `moving_jaw_so101_v1`
- **当前优先 EE link**：`grasp_center`
- `grasp_center` 是在官方 `gripperframe` site 上增加的固定 body/link，用来兼容旧版 `Genesis 0.4.0`
- `gripper` 不是夹爪指尖，而更接近腕部/夹爪基座参考点；直接拿它做 IK 目标会出现明显空间偏差
- 推荐搜索顺序更新为：
  - `grasp_center` → `moving_jaw_so101_v1` → `Moving_Jaw` → `Fixed_Jaw` → `gripper_link` → `gripperframe` → `gripper`

### 1.3 IK 姿态与 Home

```
IK_QUAT_DOWN = [0.0, 1.0, 0.0, 0.0]   # 夹爪朝下（wxyz）
HOME_DEG     = [0, -30, 90, -60, 0, 0]  # 肩→夹爪，单位°
```

- Home 跟踪误差：MJCF 0.12°（URDF 8.27°，改善 69x）
- 无需 probe quaternion search，`[0,1,0,0]` 已验证可用

### 1.4 坐标系与单位

- 除非特别说明，本文里的位置都默认是 **Genesis 世界坐标系**，单位是 **米（m）**
- 可以把它理解成：
  - `x / y`：桌面平面内的位置
  - `z`：竖直向上的高度
- `cube_pos = [x, y, z]` 表示方块**中心点**在 Genesis 世界坐标系里的位置
- `approach_z`、`offset_x / offset_y / offset_z` 不是"绝对世界坐标"，而是**相对于方块中心的偏移量**
  - 脚本里的 approach 目标可近似理解为：
  - `target_pos = cube_center_world + [offset_x, offset_y, offset_z] + [0, 0, approach_z]`
- 因此：
  - `approach_z=0.012` 的意思不是"世界坐标 z=0.012m"
  - 而是"抓取目标点位于方块中心上方 `0.012m`（12mm）"
- `cube_lift_delta` 说的是方块在 episode 中**世界坐标 z 高度的变化量**
  - 例如 `cube_lift_delta=+0.0018m`，意思是方块最终只被抬高了 `1.8mm`
- 角度默认用 **度（deg）**
  - `HOME_DEG`、`gripper_close=45` 都是角度，不是距离
- 四元数默认写成 **`wxyz`**
  - 例如 `IK_QUAT_DOWN = [0, 1, 0, 0]`
- 本文调参讨论默认站在 **Genesis 仿真坐标系** 上说；`LeRobot` 在这里主要负责数据组织，不提供另一套抓取坐标定义

### 1.5 相机与工作区

- 顶视 + 侧视同时保留，确保能观察夹爪-目标相对位置和抓取过程
- 目标物初始采样建议在前方舒适区：
  - `x ∈ [0.12, 0.20]`，`y ∈ [-0.05, 0.05]`
  - 避免贴近基座

TODO: 相机外参需要调吗?

### 1.6 控制参数

```
KP = [500, 500, 400, 400, 300, 200]
KV = [ 50,  50,  40,  40,  30,  20]
```

### 1.7 Genesis 里的 SO-101 参考

- `Genesis` 主仓库里目前**没有现成的 SO-101 抓取 demo**
- 已确认的参考只有一个基础物理测试：
  - `Genesis/tests/test_rigid_physics.py`
  - 该测试会通过 `get_hf_dataset(pattern="SO101/*")` 加载 `SO101/so101_new_calib.xml`
  - 它验证的是：
    - MJCF 能正常加载
    - 碰撞检测正常
    - box 与 robot 接触后不会数值爆炸
- 也就是说，`Genesis` 官方目前给出的 SO-101 参考更接近**稳定性 smoke test**，而不是可直接复用的 pick/grasp 参考实现

### 1.8 grasp_center / TCP 的正确理解

- `grasp_center` 可以理解为"真正希望对准目标物中心的那个参考帧"
- 对夹爪类末端来说，它通常不是：
  - 腕部 link 原点
  - 单侧手指/jaw 的 link 原点
- 它更接近：
  - 两个夹指之间的 pinch center
  - 或者工具中心点（TCP, tool center point）

## 4) Trial 与 Full Episode 的差异

`3_grasp_experiment.py` 会先跑短 trial 做 offset 搜索，再跑 full episode 做正式验证。

当前更准确的理解是：

- trial 适合快速筛掉明显无效的 offset
- full episode 才能判断"是否真的形成稳定夹持"
- **trial 变好不等于根因已解决**
- 在本项目里，早期的 trial/full-episode gap 一度误导了分析；后续确认，真正的结构性根因首先是 **EE/TCP 定义错误**

因此最佳实践是：

1. 确保 `grasp_center/TCP` 定义正确
2. 再用 trial 搜索 `offset`
3. 最后只用 full episode 判断是否真的抓起

---

## 5) 自动调参工具

`3_grasp_experiment.py`（MJCF 版）支持三维 offset 网格搜索：

```bash
python 3_grasp_experiment.py \
  --exp-id E23_grasp_center_hold2 \
  --auto-tune-offset \
  --approach-z 0.012 \
  --gripper-close 45 \
  --close-hold-steps 50 \
  --offset-x-candidates=-0.02,-0.01,0.0,0.01,0.02 \
  --offset-y-candidates=-0.02,-0.01,0.0,0.01,0.02 \
  --offset-z-candidates=-0.012,-0.010,-0.008,-0.005,0.0
```

自动化逻辑：

1. 固定同一初始 box 位姿
2. 遍历 `(ox, oy, oz)` 候选（grid search）
3. 每个候选跑短 trial（~72 步），计算 `lift_delta`
4. 选择 `lift_delta` 最大的 offset
5. 用最优 offset 跑正式 episode，输出 `.rrd + metrics.json`

> 当前经验：在 `grasp_center` 已正确定义后，`approach_z=0.012` 附近
> 比 `0.01` 或 `0.015` 更接近有效窗口。**先调高度，再调闭合。**

---

## 6) 判定逻辑（Metrics & Rules）

### 6.1 基础指标

- `cube_z_before_close`：进入 `close` 阶段前 box 的高度
- `cube_z_after_lift`：`lift` 阶段末 box 的高度
- `cube_lift_delta = cube_z_after_lift - cube_z_before_close`

### 6.2 抓取成功判定

> **注意：`delta_z` 只适合做最终批量评估的粗筛指标，不适合在 debug 阶段作为主要判据。**
> 在 debug/collision 排查阶段，`delta_z` 无法区分"被稳定夹住后抬起"和"被弹飞后落回"等完全不同的物理过程。
> 当前阶段应以 dense PNG 逐帧目视为准，重点看接触形态。

| 等级 | 条件 | 含义 |
|------|------|------|
| **成功** | `delta > 0.01m` | 方块被稳定抬起 |
| **临界接触** | `0.002m < delta ≤ 0.01m` | 已接触但夹持不稳 |
| **未抓取** | `delta ≤ 0.002m` | 未产生有效接触 |

### 6.3 辅助检查项

- `state/action` 是否在关节限位内
- close 后是否出现明显抖动或反弹
- side 视图中 box 是否进入两爪中间（vs 外侧擦边）
- `.rrd` 中 `object/cube_z` 曲线是否在 lift 阶段上升

---

## 7) 常见失败模式

| 现象 | 根因 | 对应动作 |
|------|------|---------|
| box 完全不动 | TCP 已对齐，但接触高度窗口不对 | 围绕 `approach_z=0.012` 细扫，并联动微调 `offset_z` |
| Δz 在 0.002~0.01 之间 | 已建立接触，但夹持不稳 | 先增加 `close_hold_steps`，再谨慎微调 `gripper_close` |
| trial 有正向 lift，但 episode 失败 | trial 只能筛 offset，episode 才能验证稳定夹持 | 先确认 `grasp_center` 正确，再固定高度并增加 hold |
| box 在单侧爪外缘 | 横向偏差主导 | 调 `offset_y` |
| box 在爪前方被推走 | 纵向接近太深 | 减小 `offset_x` 或增大 `gripper_open` |
| 动作幅度异常大 | IK 目标超出可达域 | 检查 cube 采样范围是否在舒适区内 |

---

## 8) 结果记录模板

每次实验至少保存：

- `grasp_<exp_id>.rrd`
- `metrics.json`（含 search_log）

`metrics.json` 推荐字段：

```
exp_id, model, xml_path, ee_link, ik_quat, home_deg,
gripper_open_deg, gripper_close_deg, approach_z, close_hold_steps,
selected_grasp_offset,
cube_z_before_close, cube_z_after_lift, cube_lift_delta, grasp_success,
auto_tune.best_offset, auto_tune.best_lift_delta, auto_tune.search_log
```

---

## 9) 实验结论（精简版）

### 9.1 当前确定成立的结论

1. **SO-101 必须先定义正确的 TCP / grasp center，再谈抓取调参。**
   - `gripper` 不能直接作为抓取参考点。
   - `moving_jaw_so101_v1` 比 `gripper` 更接近夹爪，但仍不等于真正的 pinch center。
   - 最稳定的做法是在官方 `gripperframe` site 上增加一个固定 `body/link`，命名为 `grasp_center`。

3. **`grasp_center` 路线已经验证结构正确。**
   - 在 4090 上，`grasp_center` 的 IK sanity 误差约 `0.0002m`。
   - 这说明"EE/TCP 定义错误"这个主冲突已经解决。


### 9.2 关键里程碑

| 实验 | 当前应记住的结论 |
|------|------------------|
| E20-E24 | 进入真正抓取调参阶段；`approach_z=0.012` 附近优于 `0.01/0.015`，`hold` 增加有帮助，但"更大闭合角"不一定更好 |
| E30 | 双侧斜视角 + 中间稳区采样 + auto-tune-offset | 跟新camera pose 方便debug |
| E36 - 37 | 独立 gripper 标定：扫 `-20,-10,0,10,20,30,40,50` 并导出 PNG | 已确认 **gripper 数值越大，夹爪张口越大**；之前把正值当"close"在物理上是错误的 |
| E38-E41 | 固定 `E33` 的 pose/几何后，测试 `gripper_close=-20,-10,0` | 三者 episode 都约 `+0.0006m`；gripper_close 方向问题已经从"未知"变成"已知"，但抓取失败的主因仍然是 TCP 目标点没有把 box 放进 pinch 区域。|
| E42-M1 | **XY 对齐问题确认**（见 9.4）：jaw 尖端可达 cube 高度，但 grasp_center XY 偏移导致 cube 在 jaw 外缘空抓；Z 方向可达（之前 body origin 测量误导） |
| E45 | 固定 `v3 grasp_center` 后先试 `gripper_open=10` | 比 `open=0` 更接近"包裹 box"的方向；但仍然是一侧 jaw 先碰到 box，随后发生挤压/推移，说明开口方向是对的，但幅度还不够 |
| E47-E49 | 继续验证 `open=15`、`open=15 + close=-10`、`open=15 + close=-10 + pre-close-steps=4` | `open=15` 已基本证明开口空间足够；当前更主要的问题是 `approach` 阶段的 `offset_x/offset_y` 仍让 box 偏在单侧 jaw 一边。`close=-10` 没有改变主导现象，`pre-close-steps=4` 也不再是当前优先方向 |

### 9.4 当前阻塞（E42–M1）

#### A. 实验观察（事实）

**观察 1：IK 精度没问题**

`grasp_center` 的 IK 求解精度为 0.0002m（0.2mm），可以精确到达指定目标位置。

**观察 2：link body origin 与 jaw mesh tip 不是同一个东西**

`diag_link_positions.py` 测量的是 link body origin（坐标系原点），不是 mesh 尖端：

| link | world z (body origin) | 与 cube center 差 |
|------|---------|-------------------|
| grasp_center | 0.015 | 0（IK 精度 0.0002m） |
| gripper (body origin) | 0.104 | +0.089 |
| moving_jaw (body origin) | 0.090 | +0.075 |

`diag_gc_sweep.py` 中的 `jaw_mid_z` 也是 body origin 中点，不是 mesh tip。

**观察 3：jaw 尖端确实可以到达 cube 高度**

E41/E42/E44 的 close 阶段 debug PNGs 清楚显示：jaw mesh 尖端远低于 body origin，**实际可达 cube 所在的 z=0.015 高度**。

**观察 4：cube 一直在 jaw 外缘，不在两爪中间**

回看 E41/E42/E44 close 阶段密集帧：

- E42 f059_approach → f065_close → f090_close_hold：jaw 尖端与 cube 同高，但 cube 在 fixed jaw **外侧边缘**
- E41 f033_close → f055_close_hold：jaw 触碰 cube，但 cube 被推歪而非夹住
- E44 f033_close → f050_close_hold：cube 被爪基座推倒

**每一帧都显示同一个现象：cube 在爪旁边，不在两爪之间。**

**观察 5：`verbose=True` 扰乱了 full-episode 初始状态**

`build_trajectory_chained_ik(verbose=True)` 内部执行了 `scene.step()`，改变了 robot/cube 位姿。trial 每次从 `reset_scene()` 干净启动，但 full episode 继承了被扰乱的状态。已修复。

**观察 6：trial 正向 Δz 不可复现**

`7_minimal_grasp.py` 对同一 offset 独立跑 10 次：0/10 > 0.01m，mean Δz = +0.0006m。auto-tune 中偶发的 0.01m+ 来自连续 trial 间残留的仿真状态累积。

---

#### B. 推理分析

**根因是 `grasp_center` 在 MJCF 中的挂载位置没有对准夹持中心点**

> 术语：**夹持中心点（jaw pinch center）** = fixed jaw 和 moving jaw 闭合时，两个指尖之间的中心接触点。

因果链条：

1. `grasp_center` body 挂载在 `gripper` body 下，local pos = `(-0.008, -0.0002, -0.098)`
2. `gripper` body 自身有旋转 `quat=(0.017, -0.017, 0.707, 0.707)`（约 90° 绕 Z）
3. 这个 local pos 经过 gripper 的旋转后，映射到 world frame → `grasp_center` 的世界坐标与**真正的夹持中心点**之间存在 XY 偏移
4. IK 精确地把 `grasp_center` 送到了 cube center（err=0.2mm）
5. 但**夹持中心点不等于 `grasp_center`**，而是在 XY 平面上偏离了 cube
6. 结果：cube 在 jaw 外缘，close 时擦边推开而非包裹夹住

简单说：**`grasp_center` 的定义位置歪了，不是 IK 的问题，也不是 Z 方向的问题。**
如果把 `grasp_center` 的 local pos 修正到真实夹持中心点(jaw pinch center)的位置，IK 就能把两爪正确地送到 cube 两侧。

### 9.5 固定 `v3 grasp_center` 后的后续实验（E45, E47-E49）

在 `v3 grasp_center` 固定后，这一轮调参的目标已经不是继续证明 TCP 对不对，而是看：

- cube 能不能在 `approach` 末期**进入两爪之间**
- 左右 jaw 是否能形成更对称的**几何包裹**
- `close` 前是否已经出现单侧提前接触和侧向推挤

因此，这一段的判断应以**包裹几何质量**为主，`lift_delta` 只能作为次要结果，不能反过来主导结论。

#### A. 实验观察（事实）

**观察 1：`open=10` 比 `open=0` 更接近正确方向**

`E45` 的 dense PNG 显示：把 `gripper_open` 从 `0` 增加到 `10` 后，`approach` 阶段 jaw 对 box 的包裹趋势明显好于 `E42_v3`。这说明此前并不是"jaw 太开"，而是**jaw 还不够开**。

但 `E45` 同时也显示：box 仍没有稳定进入两爪中间，依旧是一侧 jaw 先接触 box，随后发生挤压和侧向推移。

**观察 2：`open=15` 已经基本提供了足够的爪间空隙**

`E47` 相比 `E45` 显示：从 `open=10` 增加到 `open=15` 后，`approach` 阶段的 jaw 开口已经基本足以容纳 box 进入两爪之间。

但从 `ep00_f057_approach.png` 这类帧看，box 仍然没有落在两爪的中线附近，而是明显更贴近其中一侧 jaw。也就是说，当前主矛盾不再是"开口够不够大"，而是 **`approach` 阶段的 XY 放置仍有系统偏差**。

**观察 3：`open=15 + close=-10` 说明 close 过程中确实能形成一段"已夹住"的瞬间**

`E48` 中可以看到，close 初段 box 一度已经进入两爪之间，说明 `open=15` 配合较温和的 `close=-10`，已经足以让 jaw 在几何上形成夹持。

但随后 box 很快出现不符合直觉的穿透/穿入现象。这表明：

- `gripper_close` 会直接决定 jaw 继续向内压的目标角度
- 但"碰到 box 后能否被稳定挡住"并不只由 `gripper_close` 决定，还受 collision/contact 建模影响

#### B. 推理分析

**当前第一优先级不是继续放大 `open`，而是修正 `offset_x/offset_y`**

在 `open=15` 下，爪间空隙已经基本足够。当前更像是 box 在 `approach` 阶段被送到了两爪中线的旁边，而不是中线本身。

所以接下来最值得做的，是基于 `E47/E48` 的设置去调整 `offset_x/offset_y`，让 box 真正进入 pinch corridor。


### 9.6 基于 `E48` 的 collision 对照实验

`E48` 已经达到了"jaw 和 box 在 close 阶段持续接触"的前提条件，因此直接以它为基线研究 collision/contact 问题。

基线设置（与 `E48` 完全一致）：

- `xml = so101_new_calib_v3.xml`
- `open=15`、`close=-10`、`close_hold_steps=50`
- `approach_z=0.012`
- `cube_fixed = [0.16, 0.0, 0.015]`
- `offset = [0.008, -0.004, -0.01]`

collision 对照分组：

- `C54`: 基线复现（与 `E48` 参数一致）
- `C55`: 仅改 `frictionloss=0`（使用 `so101_new_calib_v3_nofrictionloss.xml`）
- `C56`: solver 强化（`implicitfast`, `substeps=8`, `iterations=100`, `noslip=10`, `timeconst=0.005`）

判断标准：

1. `close` 阶段 box 是否仍保持持续 jaw-box 接触（与 `E48` 一致）
2. 在持续接触的前提下，压入/疑似穿透是否明显减轻



## 10. 穿透问题的物理仿真因素


后续如果要继续研究穿透，建议顺序是：

3. 先改 jaw/cube 的 collision geom，使接触面更规则
4. 再给 jaw/cube 显式加 `friction`、`solref`、`solimp`
5. 最后再做 Genesis solver 侧对照：
   - 增大 `substeps`
   - 打开 `noslip_iterations`
   - 显式切到 `implicitfast`
   - 诊断性地试 `use_gjk_collision=True`

简单说，当前问题已经从"动作学没调好"切换成了"接触建模还不够可信"。

---

## 11. `3_grasp_experiment.py` 执行流程：trial vs full episode

### 脚本整体流程

```text
for each episode:
    1. reset scene（robot 回 home，cube 放到指定位置）
    2. 如果开了 --auto-tune-offset：
       a. 遍历 offset_x/y/z 候选网格（例如 5×5×1 = 25 组）
       b. 每组候选调用 run_trial()：
          - reset_scene()（settle_steps=20）
          - build_trajectory_chained_ik()（total_steps=90，比正式 episode 短）
          - 逐帧 step，记录 close 前后 cube_z
          - 返回 delta_z
       c. 选 delta_z 最大的候选作为 chosen_offset
    3. 用 chosen_offset 跑正式 full episode：
       - reset_scene()（settle_steps=30）
       - build_trajectory_chained_ik()（total_steps=完整长度）
       - 逐帧 step，同时录制图像
       - 导出 dense PNG
```

### trial 和 full episode 的区别

| | trial（`run_trial()`） | full episode |
|---|---|---|
| 目的 | 快速评估一组 offset 的 delta_z | 正式数据采集 + 图像录制 |
| 长度 | 90 步（短） | 完整 episode（长） |
| 图像 | 不录制 | 录制并导出 PNG |
| 在 scene 中 | 共享同一个 Genesis scene | 共享同一个 Genesis scene |
| reset | 每次 trial 前 reset（settle=20） | episode 前 reset（settle=30） |

### 已知问题：trial 残留影响 full episode

当开 `--auto-tune-offset` 时，25 组 trial 会在正式 episode 之前在同一个 Genesis scene 里执行。虽然每次 trial 和正式 episode 前都有 `reset_scene()`，但 reset 只重置了：

- robot 关节位置（`set_qpos`）
- robot 关节速度（`zero_all_dofs_velocity`）
- cube 位置和朝向
- cube 速度

它**没有重置** Genesis solver 的内部数值状态（如 constraint warm-start、contact cache 等）。这意味着：

- 不开 auto-tune 直接跑 full episode → 干净 scene，solver 从冷启动开始
- 开了 auto-tune 后再跑 full episode → scene 内部状态被 25 组 trial 的物理步"预热"过

实验验证：用完全相同的 offset `[0.008, -0.004, -0.01]`，`E48`（开了 auto-tune）和 `C54`（不开 auto-tune）在同一帧（f056）的 jaw 位置明显不同，说明 trial 历史确实影响了正式 episode 的物理轨迹。

### 实验设计建议

- 如果实验间需要做公平对照，所有组要么都开 auto-tune（走相同的 trial 历史），要么都不开
- 不能用"开了 auto-tune 的实验"和"没开 auto-tune 的实验"直接对比 dense PNG
- 长期应修复 `reset_scene()` 使其完全清除 solver 内部状态，或在每次 trial/episode 前重建 scene
