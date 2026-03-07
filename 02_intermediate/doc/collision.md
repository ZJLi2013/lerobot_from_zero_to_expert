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

## 3. 碰撞几何（collision geom）

Genesis 通过 MJCF 里的 `<geom class="collision">` 定义碰撞形状。

当前 SO-101 jaw 的碰撞几何是原始 STL mesh。这不一定"不好"，但原始 mesh 用于窄面夹持时容易带来：

- 接触点跳变（mesh 三角面片边缘不连续）
- 法向不稳定
- solver 在相邻帧之间收到矛盾的接触信息

更适合夹持的碰撞近似：`box` > `capsule` > convex decomposition > raw mesh

## 4. 接触参数（MJCF contact）

建立接触后，以下参数决定接触力的行为：

| 参数 | 作用 | 当前状态 |
|------|------|----------|
| `friction` | 接触后是否会滑走 | 未显式设置 |
| `solref / solimp` | penetration 被推回的速度和刚度 | 未显式设置 |
| `margin` | 接触生效的提前量 | 未显式设置 |
| `condim` | 接触摩擦维度（1=法向，3=法向+切向） | 默认 |
| `contype / conaffinity` | 碰撞过滤 | 默认 |

## 5. Genesis solver 参数

| 参数 | 作用 |
|------|------|
| `substeps` | 每帧内的子步数，越多越稳定 |
| `iterations` | 约束求解迭代次数 |
| `noslip_iterations` | 后处理防滑步，适合 pinch grasp |
| `constraint_timeconst` | 接触刚度，越小越硬 |
| `integrator` | 数值积分器（`approximate_implicitfast` / `implicitfast`） |
| `use_gjk_collision` | 替代碰撞检测方法，更稳但更慢 |

## 6. 排查优先级

```text
先改 jaw collision geom
    ↓
再加 friction / solref / solimp
    ↓
必要时再调 margin / condim
    ↓
最后才做 substeps / noslip / integrator / GJK 对照
```

### 6.1 补充排查项

1. **`frictionloss`**：Genesis 有已知 issue（#1569），`MJCF` 关节 `frictionloss` 可能导致碰撞异常。当前 `v3.xml` 存在 `frictionloss`，值得做最小对照。

2. **`substeps`**：最直接的时间离散稳定化手段。对窄接触、快速闭合、薄 mesh 接触面有效。

3. **`noslip_iterations`**：配合 `constraint_timeconst` 一起用，适合 manipulation/pinch 场景。

4. **jaw collision mesh**：确认当前 finger collision 是否仍是 raw mesh，是否需要改成 box/capsule/convex。这是最可能带来质变的一步。

---

## 7. collision 实验（基于 `E48` 基线）

`E48` 在 `close` 阶段确实建立了持续 jaw-box 接触（可观察到疑似穿透），是研究 collision 参数的合适基线。

基线设置（与 `E48` 一致）：

- `xml = so101_new_calib_v3.xml`
- `open=15`、`close=-10`、`close_hold_steps=50`
- `approach_z=0.012`
- `cube_fixed = [0.16, 0.0, 0.015]`
- `offset = [0.008, -0.004, -0.01]`

实验分组：

- `C54`: 基线复现（与 `E48` 参数一致，默认 solver）
- `C55`: 仅改 `frictionloss=0`（`so101_new_calib_v3_nofrictionloss.xml`）
- `C56`: solver 强化（`implicitfast`, `substeps=8`, `iterations=100`, `ls_iterations=100`, `noslip=10`, `timeconst=0.005`）

判断标准：

> **注意：debug 阶段不应以 `delta_z` 作为主要指标。**
> `delta_z` 只反映 box 最终高度变化，无法区分"被稳定夹住后抬起"和"被弹飞后落回"。
> 当前阶段的判断必须以 dense PNG 逐帧目视为准，重点看接触形态而不是数值。

1. `close` 阶段 box 是否仍保持持续 jaw-box 接触（与 `E48` 一致）
2. 在持续接触前提下，压入/疑似穿透是否明显减轻




### 下一步

先跑一组干净的基线（不开 auto-tune，无 trial 残留），确认在干净 scene 下用 E48 的 offset `[0.008, -0.004, -0.01]` 是什么表现，再决定后续 collision 对照怎么做。
