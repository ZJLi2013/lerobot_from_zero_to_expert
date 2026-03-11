# SO-101 抓取中的碰撞与接触

## 1. 当前问题

`E48` 实验中，`close` 阶段 jaw 与 box 确实建立了持续接触，但随后出现了不合理的压入（疑似穿透）。这不是"夹爪关得太紧"，而是 collision/contact 建模层面的问题。

## 2. 从 offset 到穿透的因果链

```text
grasp_center offset → IK 求解 → approach 位姿
    → jaw 碰撞几何 与 box 碰撞几何 接近
    → 碰撞检测生成 contact（位置、法向、穿透深度）
    → contact solver 根据 contact 算接触力
    → 接触力决定 box 是被稳定夹住、还是被挤走、还是被穿透
```

关键点：链条上每一环都可能是瓶颈。当前已经通过 `E48` 确认前半段（offset → approach → 建立接触）基本可用，问题集中在后半段。

## 3. 背景概念

### 3.1 碰撞几何（collision geom）

Genesis 通过 MJCF 里的 `<geom class="collision">` 定义碰撞形状。

当前 SO-101 jaw 的碰撞几何是原始 STL mesh。原始 mesh 用于窄面夹持时容易带来：

- 接触点跳变（mesh 三角面片边缘不连续）
- 法向不稳定
- solver 在相邻帧之间收到矛盾的接触信息

更适合夹持的碰撞近似：`box` > `capsule` > convex decomposition > raw mesh

### 3.2 visual geom vs collision geom

Genesis/MJCF 里，每个 body 可以同时有两种 geom：

- **visual geom**（`class="visual"`）：只负责渲染画面，物理引擎完全忽略它
- **collision geom**（`class="collision"`）：只负责碰撞检测和接触力计算，画面上看不到它

两者可以完全不重合。画面上看到 jaw "穿过" box，不一定意味着 collision geom 也穿过了——有可能 collision geom 根本没碰到 box（因为位置/尺寸不对），也有可能 collision geom 已经碰到了但在另一个位置。

### 3.3 接触参数（MJCF contact）

| 参数 | 作用 | 当前状态 |
|------|------|----------|
| `friction` | 接触后是否会滑走 | 未显式设置 |
| `solref / solimp` | penetration 被推回的速度和刚度 | 未显式设置 |
| `margin` | 接触生效的提前量 | 未显式设置 |
| `condim` | 接触摩擦维度（1=法向，3=法向+切向） | 默认 |
| `contype / conaffinity` | 碰撞过滤 | 默认 |

### 3.4 Genesis solver 参数

| 参数 | 作用 |
|------|------|
| `substeps` | 每帧内的子步数，越多越稳定 |
| `iterations` | 约束求解迭代次数 |
| `noslip_iterations` | 后处理防滑步，适合 pinch grasp |
| `constraint_timeconst` | 接触刚度，越小越硬 |
| `integrator` | 数值积分器（`approximate_implicitfast` / `implicitfast`） |
| `use_gjk_collision` | 替代碰撞检测方法，更稳但更慢 |

## 4. 排查优先级

```text
先改 jaw collision geom
    ↓
再加 friction / solref / solimp
    ↓
必要时再调 margin / condim
    ↓
最后才做 substeps / noslip / integrator / GJK 对照
```

---

## 5. 实验记录

所有实验基于 E48 基线（详见 best_practices.md `## 基线（E48）`）。

### C54b — 基线复现

- 配置：`so101_new_calib_v3.xml`，默认 solver
- auto-tune best offset：`[0.008, -0.004, -0.01]`
- 结论：成功复现 E48，确认基线可靠

### C55b — frictionloss=0

- 配置：`so101_new_calib_v3_nofrictionloss.xml`，其余同 C54b
- auto-tune best offset：`[0.000, -0.004, -0.01]`（与 C54b 不同）
- 现象：与 C54b 同一模式（穿透 → 挤走 → 空抓）
- 结论：frictionloss 不是穿透主因，但改变了最优 offset

### C56b — solver 强化

- 配置：`so101_new_calib_v3.xml` + `substeps=8, implicitfast, iterations=100, noslip=10, timeconst=0.005`
- auto-tune best offset：`[0.004, 0.004, -0.01]`（与 C54b 不同）
- 现象：与 C54b 同一模式；trial #19 出现 delta_z=0.118m 异常（box 被弹飞）
- 结论：solver 强化没改变主导模式，且在某些 offset 下产生不稳定接触力

**C54b-C56b 小结：** 三组都出现同一模式（穿透 → 挤走 → 空抓），frictionloss 和 solver 强化都没改变。问题在更上游：jaw 的 raw mesh collision geom 不能产生有效的夹持接触力。

---

### C57 — jaw collision geom 换 box primitive

- 配置：`so101_new_calib_v3_jawbox.xml`（moving_jaw 和 fixed_jaw 的 collision 从 raw mesh 换成 box primitive），其余同 C54b
- auto-tune best offset：`[0.004, 0.004, -0.01]`
- 现象：
  - close 阶段：visual jaw 仍穿过 box（因为 visual geom 没改）
  - close_hold 阶段：box **没被挤走**（留在原地）— 这是和 C54b 最大的区别
  - hold 阶段：爪子间距已小于 box
  - lift 阶段：空抓
- 分析：
  - collision box 的位置/尺寸是估算的，没有准确覆盖 jaw 内侧夹持面
  - 因此 collision box 既没挡住 cube（没产生夹持力），也没产生 raw mesh 那种不稳定侧向推力
  - box primitive 方向是对的（不再挤走），但需要调准位置才能真正夹住

---

### C58 — cube friction=1.5（raw mesh + `gs.materials.Rigid`）

- 配置：`so101_new_calib_v3.xml`（raw mesh collision），`--cube-friction 1.5`
- auto-tune best offset：`[0.008, -0.004, -0.01]`
- 现象：close 阶段仍有穿透和挤走，但 delta_z 从 C54b 的 0.0020 提到 0.0034
- 结论：friction 单独有一定改善，但不能解决 raw mesh collision 的根本问题

---

### C59 — dt=0.002（无效）

- 结论：不能只改 dt 不改轨迹步数， 需要同时大幅增加轨迹步数

---

### C60 — jawbox v2 + friction + box_box_detection（重要进展）

- 配置：`so101_new_calib_v3_jawbox.xml`（collision box 位置从 STL mesh 分析校准）+ `--cube-friction 1.5` + `box_box_detection=True`
- auto-tune best offset：`[0.004, 0.004, -0.01]`
- 现象：
  - close 阶段 box 被明显倾斜/推倒 — 首次看到 collision geom 产生可见接触力
  - close_hold 阶段 box 留在 jaw 附近，没被挤走
  - lift 阶段 box 没被夹起（掉了）
  - `before z = 0.0078`（close 时 box 被压到初始高度的一半），delta_z = 0.0069
- 结论：jawbox + friction + box_box_detection 组合是目前最有效的配置。接触力方向正确，但夹持力还不足以 lift

### C61 — C60 + GJK collision

- 配置：同 C60，加 `--use-gjk-collision`
- auto-tune best offset：`[0.004, 0.004, -0.01]`
- 现象：与 C60 基本一致，delta_z = 0.0070
- 结论：GJK vs MPR 在当前配置下几乎无差别，碰撞检测算法不是瓶颈

---

### 实验汇总

| 实验 | jaw geom | friction | box_box | GJK | delta_z |
|------|----------|----------|---------|-----|---------|
| C54b | raw mesh | 默认 | 否 | 否 | 0.0020 |
| C58b | raw mesh | 1.5 | 是 | 否 | 0.0034 |
| C60 | box prim | 1.5 | 是 | 否 | 0.0069 |
| C61 | box prim | 1.5 | 是 | 是 | 0.0070 |

---

## 6. auto-tune 动作参数调优（基于 C60 collision 配置）

当前 `10_auto_tune.py` 的实验配置:

- gripper：
  - 脚本变量分别是 `--gripper-open` 和 `--gripper-close`
  - 根据 `E36-E37` 的独立 gripper 标定，当前实验里 **gripper 数值越大，夹爪张口越大**
  - `open` 表示 approach 阶段使用的较大开口角
  - `close` 表示 close / close_hold 阶段使用的较小开口角，因此它应当 **小于** `open`
  - `10_auto_tune.py` 的脚本默认值是 `open=30`, `close=-10`

- sweep：`x,y in {-0.008,-0.004,0.0,0.004,0.008}`, `z=-0.01`


### 6.4  T3_xy_jawmetric 肉眼观察（oz 固定 `-0.01`）：

1. `ox+0.004` 这条线上，`oy` 从 `+0.004 -> 0.000 -> -0.004` 时，jaw 相对 box 的横向关系在变好；这说明当时的主要误差不是 Z，而是 Y 方向的 approach 偏心。

2. 但 `oy` 继续减到 `-0.008` 后，box 已经明显偏出 jaw corridor，说明这个方向虽然对，但步长再走就过头了。

3. `ox-0.004_oy+0.000` 这类组在 approach 阶段仍然会直接撞上或穿过 box，不应理解为 `box_box_detection` 没生效；更准确地说，是当前 target pose 让 jaw corridor 本身就穿进了 cube 的几何占位区。

4. `ox-0.004_oy-0.004` 这组之所以视觉上相对顺眼，是因为它同时满足了两件事：approach 阶段没有明显把 box 提前碰走，而且末帧 jaw corridor 还勉强能容下 cube。这也是它在新 metric 下成为唯一 `clean + clear` 组的原因。

5. “最后 approach 停的位置离开 box” 不应该单独归因于某个 phase 参数；它主要由 `offset_xyz + grasp_center + IK 轨迹` 共同决定。也就是说，approach 末帧停在哪里，本质上是 target pose 决定的，而不是相机错觉。

6. `ox-0.004_oy-0.008` 这类看起来像“box 粘在爪子上”的现象，通常说明 approach 过程中已经发生了接触并把 box 一起带着走；这类组会直接反映在较大的 `approach_contact` 上，因此不能算 `clean`。

7. `ox-0.008_oy+0.004` 这类“看上去基本包住，但略有挤压”的组，通常对应 `clearance_min` 接近 0 或已经为负：视觉上像是差一点，几何上其实已经没有足够余量。


## 7. 下一步实验概要

先澄清当前实验里 `open/close` 的数值语义：

- `open` / `close` 分别对应 `--gripper-open` / `--gripper-close`
- 在当前 SO-101 配置下，**数值越大，jaw 开口越大**
- 因此：
  - 增大 `open` = approach 阶段张得更开
  - 减小 `close` = close 阶段关得更紧
  - 增大 `close` = close 阶段不要关那么死

### 7.1 当前实验结论

基于 `G5b_o15_c-10` / `G5b_o20_c-10` / `G5b_o25_c-10` 三组对照，目前可以先下两个结论：

1. `open` 不是当前主矛盾。
   - 三组都能稳定复现 `C60` 的主导现象，且 `delta_z` 都是 `0.0069m`
   - 从 dense PNG 看，`open=15` 和 `open=20` 都已经基本提供了足够开口余量
   - 继续增大 `open`，并没有把系统带到一个新的“能包住 box”的 regime

2. 当前更核心的问题是 `approach` 阶段的 XY 落点仍然偏在单侧。
   - box 还没真正进入两爪中线，一侧 jaw 就先碰到 box
   - 这意味着主问题仍是 `offset_xy`，而不是 `open`

因此：

- `open` 暂时固定在 `20`
- 下一步优先微调 `offset_xy`
- `close` 仍然值得后续调整，但应当放在 `offset_xy` 之后

### 7.2 实验: 调 offset_xy

目标：在 `C60` 的 collision 配置下，找到更合适的 `offset_xy`，让 box 在 `approach -> close` 过程中尽量保持在两爪之间。

### 7.3 公共 baseline

- collision 配置：`so101_new_calib_v3_jawbox.xml` + `--cube-friction 1.5` + `box_box_detection=True`
- 必须保留 `--auto-tune-offset`，因为它承担 warm-up 作用
- 同时加 `--force-offset`，让最终 full episode 使用手动指定的 offset
- `open = 20`
- `close = -10`
- `offset_z = -0.010`
- `approach_z = 0.012`
- `close_hold_steps = 50`
- `cube_fixed = [0.16, 0.0, 0.015]`

offset 读图只保留两个结论：

- `oy` 主要控制 jaw corridor 在 box 上表面平面内的左右平移
- `ox` 主要控制 gripper 相对 box 的前后接近深度

### 7.4 第 1 轮：先扫 `oy`

只改：

- `ox = +0.004`
- `oy in {-0.004, -0.002, 0.000, +0.002, +0.004}`

目的：

- 先判断“左右偏移”是否主要由 `oy` 主导

观察：

- `oy` 从 `+0.004 -> -0.004` 后，left view 中 jaw corridor 明显整体横向移动
- 这说明 `oy` 的主效应是整体平移，而不是只改某一根爪子的相对位置
- 但 right view 中，内侧爪仍会先碰到 box 里侧，因此问题不只在 `oy`

对比图：

**ox0004_oyp0004**

![image](./images/offset_xy/ep00_f056_approach_ox0004_oyp0004.png)

**ox0004_oym0004**

![image](./images/offset_xy/ep00_f056_approach_ox0004_oym0004.png)

### 7.5 第 2 轮：固定 `oy`，再扫 `ox`

只改：

- `oy = +0.004`
- `ox in {+0.004, +0.002, 0.000, -0.002}`

目的：

- 在保持横向居中趋势的前提下，检查是否存在 “approach 太深” 问题

观察：

- `ox` 从 `+0.004 -> 0.000` 后，right view 中内侧爪“往里顶”的程度减弱
- 这说明 `ox` 的主效应更像是前后深浅，而不是左右居中
- 同时，left view 中 box 仍大致留在 jaw corridor 附近，没有立刻被挤出两爪之间

对比图：

**ox0000_oyp0004**

![image](./images/offset_xy/ep00_f056_approach_ox0000_oyp0004.png)

### 7.6 第 3 轮：anchor 重测

只改：

- `ox = 0.000`
- `oy = -0.004`

目的：

- 叠加前两轮各自观察到的正向趋势，验证是否能得到更可用的 `approach` 基线

观察：

- 与 `ox0004_oyp0004` 相比，这组在 `approach -> close` 过程中，box 仍基本留在两爪之间
- 但 `approach` 阶段仍出现较明显翻转，说明当前 baseline 还不能直接进入 `close` 调参
- 因此，当前问题已从“完全进不去 jaw corridor”推进到“能进入 corridor，但 approach 仍不够 quasi-static”

图示：

**ox0000_oym0004**

![image](./images/offset_xy/ep00_f056_approach_ox000_oym0004.png)


### 7.7 第4轮: approach 阶段 quasi-static 调试

经过上述几轮实验，`offset_x / offset_y` 的调试，已经大致能判断 `ox / oy` 对抓爪与 box 上表面平面相对位置的影响。当前有几组参数可以作为后续参考：

- `ox = +0.004, oy = +0.004`
- `ox = 0.000, oy = -0.004`
- `ox = -0.004, oy = -0.004`（TODO: auto-tune 给出的 best offset，当前未在 `03_grasp_experiment.py` 中验证）

前两组参数已经能让 box 在 `approach -> close` 过渡时基本保留在两爪之间，但还不能认为“已经抓到 box”。当前最显著的问题是：`approach` 阶段 box 会因为被按住或擦碰而发生较明显翻转，导致 box 上表面不再与水平面（操作台）平行。

对于这类简单 cube / box 抓取任务，更合理的目标是：`approach` 阶段尽量保持 box 本身 quasi-static。允许轻微接触，但不应出现明显翻转、明显平移或明显穿透。

因此下一轮的重点，是进一步调 `approach` 阶段相关参数，让 box 尽量居中、进入 jaw corridor，并在进入 `close` 前保持更稳定的姿态。


#### 7.7.1 反省

Genesis/Isaac/Gazebo 社区里，scripted grasp 在 approach 阶段出现"object tipping/moving before close"是已知的普遍问题。对于 scripted grasp，核心问题通常不是"offset_x/y 搜索"，而是：
  * approach geometry 正确性：grasp_center / TCP 标定是否准确
  * contact 配置：collision geom、contact margin、接触建立方式
  * approach 速度和步长：太快/步长太大，会导致接触时产生冲击力，box 翻转


当前路线的最大偏差是：**把"offset_x/y 搜索"当作主要手段，但 offset_x/y 只是一个补偿工具，不应该成为 scripted grasp 的主要调参路径**

更具体地说: 

1. offset_x/y 的定义本来就是"在 grasp_center 标定正确的前提下，做小范围微调"
2. 如果 approach 末帧 box 翻转明显，这说明的不是"需要搜更多 offset_x/y 组合"，而是说明:
  * grasp_center 标定本身仍有偏差
  * approach 速度/步长导致接触时冲击力过大
  * collision geom 或 contact solver 配置仍不够理想


需要先做的两个 double check

Check 1：approach 步长是否太激进？
Check 2：grasp_center 的标定结果，和当前图像里 TCP 实际落点是否一致？


#### 7.7.2  approach steps 实验

1. 调整 episode_length 配置了3组实验 [8, 12, 16]，其他参数同 7.6 第3轮实验。

现象如下:

**episode_length=12**

![image](images/quat_static/len12_ox000_oym0004.png)

**episode_length=16**

![image](images/quat_static/len16_ox000_oym0004.png)

显然，即使 16 lengths，box 仍然被戳到，发生了明显的翻转。结论：**approach step 粒度不是主因**


#### 7.7.3  IK 约束

1.. **IK 没有约束 gripper 姿态**
   - 脚本定义了 `IK_QUAT_DOWN = [0, 1, 0, 0]`，但所有 `solve_ik_seeded` 调用都传 `quat=None`
   - 这意味着 IK 只保证 `grasp_center` 到达目标 position，gripper 最终朝向由 solver 自由选择
   - 在姿态没约束的前提下，继续调 `offset_xy` 或重标 `grasp_center` 都不稳

2. 实验设计，把 `solve_ik_seeded` 里的 `quat=None` 改成 `quat=IK_QUAT_DOWN` 

3. 实验结论：grasp_center + quat=IK_QUAT_DOWN → IK 失败。 IK sanity 误差从之前的 0.2mm 暴增到 38.5mm。

**当前 SO-101 的 5-DOF 机构无法同时满足 grasp_center 的位置约束和 top-down 朝向约束**

![image](./images/quat_static/ik_quant_down.png)


#### 7.7.4 grasp_center 重新标定

[v4 标定实验](./calib.md)

从此以后，请使用  **so101_new_calib_v4.xml**, 对应 **v4 calib best offset [0.004, 0.000, -0.01]**

### 8. approach 阶段禁止碰撞

`v4` 标定已经回答了“`grasp_center` 应该对准哪里”这个问题，但还没有解决“jaw 应该如何接近 box 而不在 `close` 前先把它按翻”这个问题。后续实验的主目标，不应再是继续搜索更大的 `offset_x/y` 空间，而应明确切换为：**让 `approach` 阶段尽量无碰撞、或只允许极轻微且对称的接触**。

`approach` 末帧 cube 已进入两爪 corridor，但两爪内侧与 cube 之间仍保持很小的正间隙。


#### 8.1 approach-only 诊断

1. **approach-only 诊断**
   - 固定 `so101_new_calib_v4.xml`
   - 固定当前可达的 baseline（C60 同款 `open/close/approach_z/offset_z`）

**v4_calib+C60 baseline**

![image](./images/v4_offsetxy/v4_c60_baseline.png)


基于 `12_approach_only_diagnostic.py` 方向判断：

- `oy` 主要还是在做横向平移 / 居中
- `ox` 主要还是在做前后深浅 / 插入深度


#### v4_sweep_Exp1：固定 ox=+0.004，扫 oy

| # | ox | oy | oz | contact | tilt | clear | edge | depth | lat | flags |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | +0.004 | -0.006 | -0.010 | 0.00553 | 10.46° | -0.0076 | -0.0391 | -0.0256 | -0.0166 | |
| 2 | +0.004 | -0.004 | -0.010 | 0.00312 | 6.11° | -0.0305 | -0.0250 | -0.0136 | +0.0287 | |
| 3 | +0.004 | -0.002 | -0.010 | 0.00425 | 6.52° | -0.0235 | -0.0486 | -0.0365 | +0.0225 | |
| 4 | +0.004 | +0.000 | -0.010 | 0.00722 | 7.40° | -0.0125 | -0.0265 | -0.0142 | -0.0218 | |


__ox0.004_oym0.002__

* frame26, fixed-jaw 要碰到但未碰到 cube 内侧边
* frame27，fixed-jaw 未碰到cube，离开cube 内侧边；同时 moving-jaw 接近 cube 些许碰到 cube 外侧边
* frame28，fixed-jaw 离开cube 内侧边更远；moving-jaw 已经在 cube 上表面 靠右边缘 划进居中的位置了
* frame29, fix-jaw 离开cube 内侧边更远，moving-jaw 在cube 上表面(up) 靠右边缘 划到了靠内侧边的位置，即将离开 cube up 平面
* frame30, moving-jaw 外侧边 在 frame-29 大概相同的位置 按住了 cube ，cube 发生了转动 
* frame31, 32, 同frame30，cube有进一步转动

![image](./images/v4_offsetxy/f032_approach_ox0004_oym0002.png)

**approach last frame 显然 cube center 跟 jaw midpoint 在 xy 平面 有些 offset**


__oxp0.004_oym0.004__

* frame27, fixed-jaw 远离 cube 内侧边; moving jaw 接近 cube 左侧边
* frame28, fixed-jaw 进一步远离 cube 内侧边; moving jaw 沿着 cube 左侧边划到 边沿居中的位置
* frame29,  moving jaw 沿着 cube 左侧边 进一步 划到 边沿靠内侧边，快要离开cube 的状态
* frame30, moving-jaw 外侧边 在 frame-29 大概相同的位置 按住了 cube ，cube 发生了挤压
* frame31, 32, 同frame30，cube有进一步被挤压


![image](./images/v4_offsetxy/f032_approach_oxp0004_oym0004.png)

**approach last frame 显然 cube center 跟 jaw midpoint 在 xy 平面 有些 offset**


__oxp0.004_oy0.00__

![image](./images/v4_offsetxy/f032_approach_ox0.004_oy0.00.png)



#### v4_sweep_Exp2：固定 oy=-0.004，扫 ox（含负方向）

| # | ox | oy | oz | contact | tilt | clear | edge | depth | lat | flags |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | +0.004 | -0.004 | -0.010 | 0.00553 | 9.96° | -0.0080 | -0.0378 | -0.0244 | -0.0170 | |
| 2 | **+0.002** | -0.004 | -0.010 | **0.00286** | **4.47°** | -0.0293 | -0.0252 | -0.0142 | +0.0275 | **●contact ●tilt** |
| 3 | +0.000 | -0.004 | -0.010 | 0.00306 | 4.29° | -0.0239 | -0.0492 | -0.0364 | +0.0228 | ●tilt |
| 4 | -0.002 | -0.004 | -0.010 | 0.00489 | 6.62° | -0.0133 | -0.0288 | -0.0178 | -0.0228 | |


__oxp0.004_oym0.004__


* fix-jaw 在外，moving-jaw 在内
* frame28, moving-jaw 靠内侧，远离 cube 内侧边，而 fixed-jaw 快接近 cube up 平面靠左边缘
* frame29 后续, fixed-jaw 大概按住cube 了，cube 有些翻转，且后续被挤压

![image](./images/v4_offsetxy/f032_approach_oxp0004_oym0004.png)

**注意，这一组 fix/moving-jaw 相对位置不同。其他组 fix-jaw 靠内，moving-jaw 靠外**


__oxp0.002_oym0.004__

* frame27,  fix-jaw 已经离cube 内侧边有大概 5cm 甚至更多了；而 moving-jaw 大概碰到外侧边和前侧边角点
* frame28,  moving-jaw 沿着前侧边划到居中的位置，fix-jaw 完全没接触
* frame29, moving-jaw 沿着前侧边划到靠内侧边角点的位置，快离开cube up 平面了
* frame30 ~ 32，moving-jaw 按住了 cube 大概 前侧和内侧角点的位置，持续挤压cube


![image](./images/v4_offsetxy/f032_approach_ox0002_oym0004.png)

**approach last frame 显然 cube center 跟 jaw midpoint 在 xy 平面 有些 offset**


__ox0.000_oym0.004__


* fix-jaw 在内，moving-jaw 在外
* last 3 frames，moving-jaw 按住了cube 持续挤压

![image](./images/v4_offsetxy/f032_approach_ox00_oym0.004.png)

__oxm0.002_oym0.004__

* fix-jaw 在外，moving-jaw 在内
* * frame28, moving-jaw 靠内侧，远离 cube 内侧边，而 fixed-jaw 快接近 cube up 平面靠右边缘
* frame29 后续, fixed-jaw 大概按住cube 了，cube 先有些翻转，后续被挤压

**注意，这一组 fix/moving-jaw 相对位置同 oxp0.004_oym0.004**



**关键观察**：上述实验中，由于 grasp-center 和 cube-center 存在 offset，不论哪个jaw在外侧，在收到 grasp-center的路径上，都会蹭到cube 上平面从而产生挤压或者翻转。


#### v4_sweep_Exp3 使用 offset_xy={0, 0}

```py
pos_approach = cube_pos + off + np.array([0.0, 0.0, approach_z])
``` 


![image](./images/v4_offsetxy/f032_approach_ox0_oy0.png)



### 8.2 反省


1. approach 阶段某个 jaw 先撞到 cube 上表面。
2. `grasp_center` 视觉上和 `cube center` 有明显 gap。
3. 调整 `offset_x / offset_y` 会改变碰撞朝向，但碰撞模式本身没有被消除。

**Exp1/Exp2/Exp3（有 cube）数据分析**

以下值来自各实验 `summary.json` 的 `best_candidate`：

| exp | offset(ox, oy, oz) | closing_axis_offset | approach_axis_offset | centering_error | jaw_balance_error | 结论 |
|---|---|---:|---:|---:|---:|---|
| Exp1 | (0.004, -0.004, -0.01) | +0.0287m | +0.0264m | 0.0413m | 0.0665m | 偏差很大，且有 tilt |
| Exp2 | (0.002, -0.004, -0.01) | +0.0275m | +0.0269m | 0.0410m | 0.0644m | contact/tilt 改善，但几何偏差仍大 |
| Exp3(0,0) | (0.000, 0.000, -0.01) | -0.0199m | +0.0143m | 0.0332m | 0.0290m | 方向改变但偏差仍显著 |

补充观察（看 `results_ranked`）：

* `closing_axis_offset` 在不同 `ox/oy` 之间会正负翻转（例如约 `+0.02m` 到 `-0.02m`），说明 XY offset 只是在左右两侧切换。
* `approach_axis_offset` 多数仍为正，且常在厘米级，说明沿 approach 方向仍存在系统偏差。
* `centering_error` 与 `jaw_balance_error` 没有被 offset sweep 拉到很小量级（仍是厘米级）。

**关键 metric 含义**

* `centering_error`: `||cube_center_world - jaw_midpoint_world||`，表示 cube 中心与两爪中点的 3D 距离（越小越好）。
* `jaw_balance_error`: `abs(dist_to_fixed_jaw - dist_to_moving_jaw)`，表示 cube 到两爪“受力/距离”是否对称（越小越好）。
* `closing_axis_offset`: `(cube_center - jaw_midpoint)` 在 closing 轴上的投影（符号表示偏向哪一侧，绝对值越小越好）。
* `approach_axis_offset`: `(cube_center - jaw_midpoint)` 在 approach 轴上的投影（绝对值越小越好）。
* `tcp_offset`: `tcp_actual - ik_target`，表示实际 TCP 相对 IK 目标点的偏差向量；若在多组 offset 下方向/量级长期稳定，通常意味着 TCP/reference/calibration 存在系统误差。

**阶段性结论（基于有 cube 实验）**

* 仅靠 `offset_xy` 不能解决问题。
* 先前观测到的大 `tcp_offset` 不能直接等价为“固定系统 TCP 偏差”，还需和无接触基线对照。

#### 最小快速验证


**4090 实测（genesis_poc:latest，v4 baseline 三组 offset）**

| exp_id | offset(ox, oy, oz) | ik_target (x,y,z) | tcp_actual (x,y,z) | tcp_offset (x,y,z) | gripper_jaw_z | cube_top_z | jaw_below_top |
|---|---|---|---|---|---:|---:|---|
| qv_ox000_oy000 | (0.000, 0.000, -0.010) | (0.1600, -0.0000, 0.0167) | (0.1644, -0.0001, 0.0229) | (+0.0044, -0.0001, +0.0062) | 0.0649 | 0.0289 | False |
| qv_ox000_oym004 | (0.000, -0.004, -0.010) | (0.1600, -0.0040, 0.0167) | (0.1644, -0.0043, 0.0229) | (+0.0044, -0.0003, +0.0061) | 0.0650 | 0.0291 | False |
| qv_oxp004_oy000 | (0.004, 0.000, -0.010) | (0.1640, -0.0000, 0.0167) | (0.1688, -0.0002, 0.0235) | (+0.0048, -0.0002, +0.0068) | 0.0641 | 0.0296 | False |

该组快检仅说明：在“有 cube、approach 接近接触”的条件下，`tcp_offset` 可能显著放大。

### 8.3 TCP reference 系统偏差排查

目前可确认：

1. 在 `so101_new_calib_v4.xml` 中，`gripperframe` 与 `grasp_center` 的 `pos/quat` 一致。
2. `3_grasp_experiment.py` 在 `ee_link=grasp_center` 分支下，`tcp_actual` 直接取 `ee_pos`，并未出现“planner_tcp 与 sim_tcp 两套不同定义”的显式分裂。
3. 因此，“planner_tcp 与 sim_tcp 定义不同”不是当前代码路径下的直接证据。

---

**无接触基准结果**

使用 `31_tcp_nocontact_grid.py`（4090 + `genesis_poc:latest`）在无 cube 条件下：

```yml
target_x: [0.156, 0.160, 0.164]
target_y: [-0.004, 0.000, 0.004]
target_z: 0.060
points: 9
repeats: 3
```

关键统计（`v4_tcp_nocontact_v2/summary.json`）：

* `tcp_offset_global_mean = [ +0.00050, -0.00001, +0.00049 ] m`
* `tcp_offset_global_std_over_points = [ 0.00126, 0.00003, 0.00118 ] m`
* `delta_jaw_global_mean = -0.02241 m`
* `delta_jaw_global_std_over_points = 0.00259 m`

**注意** 这里 fix_jaw(gripper) 跟 moving_jaw 的 delta_z 计算是基于 两爪的 link origin 的，而不是 jaw endpoints，不过考虑 jaw link origin 跟 jaw endpoints 是刚性连接。


观察：

1. 大多数点 `tcp_offset` 接近 0（亚毫米到毫米级）；少数点会出现离群样本（例如约 `+1cm` 量级），会抬高点内 std。
2. 无接触下 `delta_jaw` 并不为 0（均值约 `-22mm`），说明“jaw 相对高度差”在无接触时也存在，不能仅用接触来解释；其来源包含 `quat=None` 姿态自由度 + link origin 定义。
3. 与有 cube 结果对照，先前大偏差更像接触扰动 + 局部 IK/姿态残差放大，而非单一固定 TCP transform 错位。

**更新结论**

* root cause 更合理的表述是：`approach` 几何导致的接触扰动为主，`grasp_center` 残余几何误差为重要放大因子之一。
* `grasp_center` 校准依然需要继续做，但不应被当作唯一根因。
* 即使 offset_xy={0,0}，jaw 在 approach 过程先与 cube 接触，导致最后帧位置/姿态相对“理想无接触到位”发生偏离。



### 9. approach 几何实验

目标：降低 approach 阶段的先碰撞概率，优先避免“先碰后夹”。

已知问题：

* 现有 `approach_z=0.012` + `offset_z=-0.010` 下，目标高度偏低，路径几何容易穿过 cube top 附近。
* `quat=None` 时末端姿态有漂移空间，同样 target 下 jaw 内侧几何位置不稳定，放大单侧先碰风险。

---

#### 9.1 ~~方案A：提高 approach_z~~

目的：先把最终 approach 停点抬高，减少 jaw tip 穿过 cube top 的概率。

建议扫描：

```yml
approach_z: [0.020, 0.024, 0.028, 0.032]
offset_z:   -0.010 (先固定)
```

![image](./images/v4_approach_z/f032_approach_approachz0.032.png)

显然，单纯提高 approach-z 只是在当前阶段避免了接触cube，但是从几何上看，接下来一定还是会擦/按到cube。所以调整 approach-z 应该意义不大。



---


### 10. 轻量脚本实验 

参考点定义（统一口径）：

- `grasp_center`：IK 目标点（`link=grasp_center`），用于控制末端到达。
- `link origin`（`gripper` / `moving_jaw_so101_v1`）：仅是各 link 坐标原点，不等于接触点。
- `jaw inner surface`：由 XML 中 `fixed_jaw_box` / `moving_jaw_box` 推导出的内侧接触面中心（推荐几何参考）。
- `jaw_midpoint`：两侧 `jaw inner surface` 的中点，优先作为“夹持几何中心”。
- `error_center_minus_midjaw`：`grasp_center - jaw_midpoint`（当前重点观察其 Z 分量）。

- 在 Genesis 里，link.get_pos() 返回的是 link frame origin 的世界坐标，不是接触点/夹爪尖端点
  * （Genesis/genesis/engine/entities/rigid_entity/rigid_link.py 的 get_pos() 注释就是 world-frame link position）

- 当前 XML（so101_new_calib_v4.xml）里明确有两块专门的接触几何
  * fixed_jaw_box
  * moving_jaw_box 
  # 注释直接写的是 inner pinch surface（内侧夹持面）。

- 12_approach_only_diag.py 中的使用正确:
  * 从 XML 读 fixed_jaw_box / moving_jaw_box，计算各自的 inner surface world point，再算 jaw_midpoint 
  * 这比直接使用 gripper/moving_jaw link origin 更接近真实加持参考

- jaw 参考点不应该默认时 link origin ，而应该是 inner pinch surface midpoint(接触面中点)
- 对于通用抓取几何标定，优先选 **接触面中心**；除非是明确**抓尖抓取(tip grasp)**，才考虑用指尖点(tip point)



### 10.1 调平delta_z 实验

目标：将 `delta_z = z_moving - z_fixed` 收敛到阈值内（当前阈值 `|delta_z| <= 0.004m`），再进入稳定抓取。

当前在 `33_grasp_light.py` 上验证了两种“单变量调平”策略：

1. **策略A：仅调航向角（yaw-only）**
   - 做法：在 gate 触发后，只扫描 `yaw`（先粗扫，再细扫），其他量不变。
   - 结果（Exp3 参数）：`dz_before=-0.03447m -> best_dz=-0.02574m`，改善约 `8.7mm`。
   - 判定：仍远大于 `0.004m`，未通过 gate（`success_count=0`）。

2. **策略B：仅调 roll 角（roll-only）**
   - 做法：仅扫描 `roll`（`±2°` 粗扫到 `±16°`，命中后 `1°` 细扫）。
   - 结果（Exp3 参数）：`dz_before=-0.03447m -> best_dz=-0.02574m`，改善量与 yaw-only 接近。
   - 判定：同样未收敛到阈值内（`success_count=0`）


图示（概念）：

```text
世界坐标系（示意）:
        z ^
          |
          +----> x
         /
        y

航向角 yaw: 围绕 z 轴旋转（平面内“转向”）
            z ^
              |   ↻ yaw
              o------> x

roll 角: 围绕末端前进轴旋转（“左右倾斜”）
          爪子连线一侧上/下倾，直接影响两爪高度差 delta_z
```



3. 结论（补充 no-cube 参考系验证）：

| strategy | best_dz  |
| -------- | -------- |
| yaw      | -0.02574 |
| roll     | -0.02574 |

`yaw-only` 与 `roll-only` 收敛到几乎同一 `best_dz`，说明单自由度姿态调平能力受限，疑似存在参考系偏置。

在 no-cube 场景（`34_nocube_reference_check.py`）下进一步验证：

- `error = z_grasp_center - (z_fixed + z_moving)/2`
- `approach` 阶段均值：
  - `quat_mode=none`：约 `-0.0636 m`
  - `quat_mode=pregrasp_flatten_yaw`：约 `-0.0622 m`

这表明即使无接触，`grasp_center` 与 jaw midpoint 仍存在厘米级系统偏差；当前问题更像 **reference/frame 定义问题优先**，而非仅靠 yaw/roll 控制可解。


4. `34_nocube_reference_check`结论整理

- 推荐口径（inner surface）：
  - 采用`link origin`参考，“约 6cm 偏差”，但不代表真实夹持接触中心。
  - 采用 `fixed_jaw_box/moving_jaw_box` 内侧接触面中点后，`grasp_center` 与 jaw midpoint 在 `approach` 基本重合。
- 结论：
  - 后续标定与诊断统一使用 `inner-surface midpoint`；
  - `link origin` 指标仅作历史对照，不再作为主判据。




  

### 10.2 

其实从现在的pngs 中可以看到，即使调平 Jaws 的 delta-z ，jaws mid-point 水平方向跟 cube center xy 还是有 gap。这个大概需要offset_xy 调整 ？




