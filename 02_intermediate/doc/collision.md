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

> **注意：** debug 阶段不应以 delta_z 作为主要指标，需看 dense PNG。

### C54b — 基线复现

- 配置：`so101_new_calib_v3.xml`，默认 solver
- auto-tune best offset：`[0.008, -0.004, -0.01]`
- 现象：close 阶段穿透 → close_hold 阶段 box 被挤走 → hold 爪子间距很小 → lift 空抓
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

### C58 — cube surface friction=1.5（失败）

- 配置：同 C54b，加 `--cube-friction 1.5`
- 结果：Genesis `gs.surfaces.Default` 不接受 `friction` 参数，报错退出
- 原因：Genesis 的 surface 只管渲染，物理 friction 需要通过 MJCF geom 属性设置
- 结论：GPT 给的 API 是错的。如果要改 cube friction，需要把 cube 定义成 MJCF entity 而不是 `morphs.Box`

### C59 — dt=0.002（无效）

- 配置：同 C54b，加 `--sim-dt 0.002`
- 结果：机械臂完全没到 box 附近，所有 trial delta_z=0.0000
- 原因：dt 从 0.033 降到 0.002，但轨迹步数仍按 `episode_length * fps = 240` 算。每步物理时间缩短 16 倍，8 秒内机械臂走不完轨迹
- Genesis 还警告：dt<2ms 在 `use_gjk_collision=False` 下可能数值不稳定
- 结论：不能只改 dt 不改轨迹步数。GPT 建议的 dt=0.002 需要同时大幅增加轨迹步数，当前脚本结构不支持

---

### C58b — cube friction=1.5（raw mesh + `gs.materials.Rigid`）

- 配置：`so101_new_calib_v3.xml`（raw mesh collision），`--cube-friction 1.5`
- auto-tune best offset：`[0.008, -0.004, -0.01]`
- 现象：close 阶段仍有穿透和挤走，但 delta_z 从 C54b 的 0.0020 提到 0.0034
- 结论：friction 单独有一定改善，但不能解决 raw mesh collision 的根本问题

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

## 6. 动作参数调优（基于 C60 collision 配置）

固定 collision 配置：`v3_jawbox.xml` + `--cube-friction 1.5` + `box_box_detection=True`

目标：让 approach 末帧 box 处于两爪较中间的位置。

判断标准：只看 approach last frame 的 dense PNG，box 是否在两个 jaw 之间，而不是贴在单侧。

实验设计：

固定 C60 collision 配置 + `open=20, close=-20, approach_z=0.012`，不开 auto-tune，手动指定 offset，逐组看 approach 末帧。

offset_z 固定 `-0.01`，扫 x/y 的 3x3 网格（围绕 auto-tune 选出的 `[0.004, 0.004]`）：

| 组 | offset_x | offset_y |
|----|----------|----------|
| A1 | 0.000 | 0.000 |
| A2 | 0.000 | 0.004 |
| A3 | 0.000 | 0.008 |
| A4 | 0.004 | 0.000 |
| A5 | 0.004 | 0.004 |
| A6 | 0.004 | 0.008 |
| A7 | 0.008 | 0.000 |
| A8 | 0.008 | 0.004 |
| A9 | 0.008 | 0.008 |

注意：不开 auto-tune 时需要处理 Genesis warm-up 问题（见 best_practices.md）。可能需要先跑一轮 auto-tune warm-up 然后在同一 scene 里按固定 offset 跑，或者接受 warm-up 差异只做相对对比。
