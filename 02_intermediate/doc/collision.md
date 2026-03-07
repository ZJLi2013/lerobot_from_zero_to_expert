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

### 下一步

把 collision box 的 pos/size 调到真正覆盖 jaw 内侧夹持面：

1. 从 visual mesh 顶点数据读出 jaw 内侧面的实际位置范围
2. 或用 Genesis geom 可视化（`group=3`）直接看 collision box 位置
3. 迭代调整直到 collision box 和 visual jaw 内侧面大致重合
