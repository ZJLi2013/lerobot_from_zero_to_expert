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

**B1. Z 方向可达**

**B2. 根因是 `grasp_center` 在 MJCF 中的挂载位置没有对准夹持中心点**

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


#### 9.4.5 当前建议

> **核心问题是 grasp_center 的 XY 偏移未对准 jaw pinch center，导致空抓。Z 方向可达。**

后续推荐方向（按优先级）：

1. **标定 grasp_center → jaw pinch center 的 XY 偏移量**：在仿真中固定 cube 位置，用 `offset_x / offset_y` 的精细网格搜索（步长 1-2mm）找到让 cube 正好落入两爪中间的补偿值。搜索范围建议 `ox ∈ [-0.03, 0.03]`, `oy ∈ [-0.03, 0.03]`
2. **修正 MJCF 中 `grasp_center` 的 pos**：根据标定结果，将 `grasp_center` body 的 pos 从当前值调整到实际 jaw pinch center 对应的 local offset

---



