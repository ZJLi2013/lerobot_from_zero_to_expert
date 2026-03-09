# SO-101 `grasp_center` 标定说明

## 公共信息

**目标**：把 MJCF 里 `grasp_center` 这个 TCP frame 调整到更接近真实 jaw pinch corridor 的局部定义。

**标定流程**：
1. 固定 cube 在 `[0.16, 0.0, 0.015]`，固定抓取参数（`approach_z=0.012`、`close_hold_steps=50`）
2. 在世界坐标系里对运行时抓取目标加一组 XY offset，遍历候选
3. 按评分函数选出最优 offset，把世界坐标系下的补偿反推到 `gripper` 局部坐标系，写回 XML

**`grasp_center.pos` 默认值（default）**：`[-0.0079, -0.000218121, -0.0981274]`

**`grasp_center.quat` 当前值（v3 XML）**：沿用现有 XML 定义，尚未系统重标

**当前使用版本**：`v3` → `so101_new_calib_v3.xml`

---

## 各版本对比

| 版本 | `grasp_center.pos` local | 评分函数逻辑 | E42 主要现象 | 结论 |
|---|---|---|---|---|
| `default` | `[-0.0079, -0.000218, -0.0981]` | 官方默认，无修正 | XY 常落在 box 边缘，close 时容易擦边空抓 | 可达性好，但 TCP 未对准真实夹持中心 |
| `v1` | `[-0.0297, -0.0039, -0.0957]` | 优先 `lift_delta` / `close_contact_delta` | 明显 over-shot；从一侧偏到了另一侧 | 旧评分把"推着走"误判为好结果 |
| `v1_half` | `[-0.0188, -0.0021, -0.0969]` | `default -> v1` 修正量取 50% | 比 v1 更合理，但仍未进入两爪中心，close 时仍推走 box | 方向对，但还差一点 XY 微调 |
| `v2` | `[+0.0004, +0.0069, -0.1023]` | 优先 close 首帧 `local_xy_center_error` | XY 投影更居中，但 IK sanity 退化到 ~7.9mm，lift=0 | 几何评分好但 IK 可达性退化，实际执行更差 |
| `v3` | `[+0.0093, -0.0065, -0.0942]` | 先约束 approach 可达性，再考虑 close 居中 | IK sanity 恢复 0.2mm；lift_delta=+0.004m；close 阶段仍有挤走 | 当前最可用；下一步优先调抓取流程参数 |

---

## 关键数值

| 版本 | `selected_grasp_offset` | `cube_lift_delta` | `IK sanity err` |
|---|---|---:|---:|
| `v1` | `[+0.008, +0.004, -0.01]` | `+0.00048 m` | ~0.2 mm |
| `v1_half` | `[+0.004, -0.008, -0.01]` | `+0.00053 m` | ~0.2 mm |
| `v2` | `[-0.004, +0.008, -0.01]` | `+0.00000 m` | ~7.9 mm |
| `v3` | `[+0.004, +0.004, -0.01]` | `+0.00401 m` | ~0.2 mm |

---

## `8_calibrate_grasp_center.py` 当前能力边界

当前 `8_calibrate_grasp_center.py` 的实际流程可以简化为：

1. 读取 XML 里的当前 `grasp_center.pos` / `grasp_center.quat`
2. 固定 cube pose，做 deterministic XY offset coarse + refine 搜索
3. 用 `approach_tcp_error`、`approach_z_abs_error`、`approach_xy_error`、`local_xy_center_error` 等指标排序
4. 输出：
   - best runtime offset（world）
   - `suggested_grasp_center_delta_local`
   - `suggested_grasp_center_pos_local`

当前脚本的能力边界：

- 会读取 `grasp_center.quat`
- 但运行时 IK 仍是 `quat=None`
- 最终只会给出新的 `grasp_center.pos`
- **不会搜索或输出新的 `grasp_center.quat`**

所以，当前脚本本质上仍是 **TCP 位置标定脚本**，不是完整的 **TCP frame（`pos + quat`）标定脚本**。

---

`v3` 的 `grasp_center.pos` 仍是当前最可用版本，但后续collison实验说明：问题已经不再只是流程参数或 runtime `offset_xy`。当前更准确的判断是：

1. `offset_xy` 只能做小范围补偿，不能替代 TCP frame 本身的定义
2. 单纯把 `quat=IK_QUAT_DOWN` 硬塞给当前 `grasp_center` 不可用
3. 现有 `8_calibrate_grasp_center.py` 只标定 `pos`，还没有真正验证/重标 `grasp_center.quat`


## v4 标定设计

### 验证 TCP 轴语义

现在的问题，已经不是单纯的 pos 偏了一点，而更像是：

   * grasp_center 这个 frame 的轴方向本身就不一定对
   * 如果 frame basis 定义错了，直接搜 quaternion 只是把错误坐标系继续旋转

* 可视化 grasp_center frame 三轴
   * grasp_center 的局部 x/y/z 轴在世界系里分别朝哪边
   * 它和 fixed jaw / moving jaw 的真实几何关系是否一致
   * 当前 grasp_center.quat 是不是根本没对准 jaw pinch corridor

### 几何重建 grasp_center.quat

* 用 jaw corridor / closing / approach 三条几何方向构造 rotation matrix
* 再转 quaternion
* 不做 brute force quat 搜索

### 残余 pos 微调

* 在新的 frame basis 上，再用现有 8_calibrate_grasp_center.py 思路做小范围 pos 搜索
* 这时 offset_xy 才回到它本来的角色：小补偿


### 实验结果及分析

2026-03-09 重新用 `11_grasp_center_v4.py` 做了两组对齐后的诊断实验，并在 4090 + `genesis_poc:latest` 上复跑：

1. `C60-like`
   - 配置对齐到 `collision.md` 里的 `C60` 家族 baseline：
     - `so101_new_calib_v3_jawbox.xml`
     - `cube-friction=1.5`
     - `box_box_detection=True`
     - `open=20`, `close=-10`, `close_hold_steps=50`
     - `approach_z=0.012`, `offset_z=-0.010`
     - 保留 warm-up auto-tune 历史
   - 在当前 v4 诊断脚本的简化 warm-up 复现里，auto-tune 选出的 offset 是 `[0.008, 0.000, -0.010]`，和 `collision.md` 里记录的 C60 `[0.004, 0.004, -0.010]` 不完全一致，但仍属于同一组 collision / warm-up baseline。

2. `7.6 第三轮 anchor`
   - 在相同 baseline 上，加 `force-offset`
   - 最终固定 offset 为 `ox=0.000, oy=-0.004, oz=-0.010`
   - 这一组与 `collision.md` `7.6 第 3 轮` 的目标配置一致

两组实验得到的几何结论几乎完全一致：

- 当前 `grasp_center` local:
  - `pos = [0.009342, -0.006544, -0.094165]`
  - `quat = [0.707107, 0.0, 0.707107, 0.0]`
- 几何重建后建议的 `grasp_center` local:
  - `pos ≈ [0.013867, 0.002400, -0.082094]`
  - `quat ≈ [0.144190, -0.598573, -0.684636, -0.390119]`
- 轴语义误差（`x/y/z`）在两组里都稳定在：
  - `x ≈ 131.64 deg`
  - `y ≈ 91.20 deg`
  - `z ≈ 74.36 deg`

#### 稳定性与局限

- `C60-like` 和 `7.6 anchor` 虽然最终 `offset_xy` 不同，但导出的 `suggested_grasp_center.pos/quat` 基本一致
- `residual_local_delta` 在两组里也几乎完全相同，约为 `[+0.004525, +0.008944, +0.012071]`

- 两组一致的根本原因：展开推导可以证明 `residual_local = jaw_mid_local(gripper_angle) - gc_local_pos`，其中 `gripper_world_rot` 完全约掉。也就是说 **`residual_local_delta` 只取决于 `gripper_angle` 和 `gc_local_pos`，与 arm 构型（肩/肘/腕关节角度）无关**。两组用的是同一份 v3 XML 和同一个 `gripper_open=20`，所以残差完全一致
- **不存在循环依赖**：写入 v4 pos 后 IK 确实会把手臂送到新构型，但 jaw midpoint 在 gripper 局部系中的位置只取决于 gripper joint angle，不受上游关节影响。因此 v4 的 jaw-midpoint 方法预期一步收敛：写入 XML 后再验证，`residual_local_delta` 应接近 `[0, 0, 0]`
- **v1 → v2 → v3 每轮结果不同的真正原因**：不是循环依赖，而是每一轮用了不同的评分函数、不同的 offset_z、不同的参考对象（见 diff 来源表格）

#### 关于 quat 误差的正确解读（图示）

先看两张 C3 标定实验（`8_calibrate_grasp_center.py`）导出的 close-phase 截图：

**C3 rank01** — 标定评分最优组（`ox=+0.008, oy=+0.017, center_err=0.0088`）：

![C3_rank01](./images/v4_calib/C3_rank01_close.png)

**C3 rank03** — 第三名（`ox=+0.010, oy=+0.017, center_err=0.0235`）：

![C3_rank03](./images/v4_calib/C3_rank03_close.png)

两张图里 cube 都明显在 jaw 外缘而不在两爪中间。即使 rank01 的 `center_err` 已经是 `0.0088`（gripper 局部系里 cube 与 gripper 原点的 XY 距离），jaw 尖端仍然只是擦过 cube 侧面。rank03 更差：jaw 已经绕过 cube 开始向另一侧倾斜。这就是 `8_calibrate_grasp_center.py` 在 pos-only 标定下的典型结果。v1 → v2 → v3 三轮给出的 delta 各不相同，不是因为"循环依赖"，而是因为每轮用了不同的评分函数和 offset_z。

接下来看 `11_grasp_center_v4.py` 的轴诊断图：

**V4 诊断图**（7.6 anchor 配置，`ox=0.000, oy=-0.004`，实线/虚线版）：

![V4_anchor76](./images/v4_calib/V4_anchor76_frame_diag.png)

右侧两个投影面板（Top View XY / Side View XZ）读法：

| 图元 | 含义 |
|------|------|
| 黑色连线 | 两个 jaw box inner surface 之间的连线（fixed jaw → moving jaw） |
| 品红实心点 | jaw midpoint = 两个 inner surface 的中点，即"理想 TCP 位置" |
| 红色方块 | cube center `[0.16, 0.0, 0.015]` |
| 蓝色实心点 | 当前 `grasp_center` 在世界系中的实际位置 |
| 灰色实心点 | `gripper` body 的世界系位置（腕部/基座） |
| **实线箭头**（红/绿/蓝） | 当前 `grasp_center` frame 的 x/y/z 轴，原点在蓝点处 |
| **虚线箭头**（红/绿/蓝） | 从 jaw corridor 几何重建的 x/y/z 轴，原点在品红点处 |

颜色统一：**红 = x，绿 = y，蓝 = z**。

从图中可以读出两个信息：

1. **位置残差**：蓝点（current grasp_center）和品红点（jaw midpoint）之间的间距 ≈ `residual_world_delta`。在 Top View 里蓝点在品红点左侧偏下，Side View 里蓝点在品红点左下方，说明当前 `grasp_center` 相对于真实夹持中心在 XY 和 Z 上都有偏差。

2. **轴方向不一致**：实线箭头（current frame）和虚线箭头（rebuilt frame）完全不重合。Top View 里最明显的是红色箭头（x 轴）：实线红指向左侧，虚线红指向右上，基本相反（对应 `x ≈ 131.64 deg`）。绿色箭头（y 轴）也近乎正交（对应 `y ≈ 91.20 deg`）。

**但关键问题是：这个轴误差在当前 pipeline 里有没有实际影响？**

轴语义误差 131/91/74 度是真实的测量值，说明当前 `grasp_center` frame 的朝向确实和 jaw corridor 几何不一致。但需要区分两件事：

1. **当前 pipeline 里 quat 不影响抓取结果**
   - 所有 IK 调用都是 `quat=None`（position-only IK）
   - `grasp_center.quat` 不参与 IK 求解，不改变手臂构型
   - `residual_local_delta` 的计算用的是 `gripper_world_rot`，不是 `gc_world_rot`，因此 pos 修正完全不依赖 `grasp_center.quat` 的值
   - v1–v3 历史标定（`8_calibrate_grasp_center.py`）同理，也不依赖 `grasp_center.quat`

2. **quat 修正的价值是语义正确性，不是当前 blocker**
   - 如果未来需要 orientation-constrained IK（例如强制 top-down 抓取），正确的 `grasp_center.quat` 是前提
   - 但对于 5-DOF SO-101，位置 + 姿态同时约束本身就超出自由度限制（collision.md 7.7.3 已验证 IK_QUAT_DOWN 失败）
   - 所以 quat 修正是"为未来铺路"，不是"不修就不能继续"

#### 结论

1. **pos 是当前的直接可操作项**：suggested_local_pos `≈ [0.0139, 0.0024, -0.0821]` 相对 v3 `[0.0093, -0.0065, -0.0942]` 是一个量级为 ~12mm 的修正，可以直接写进 XML 生成 `v4` 候选，再通过 full-episode 实验验证
2. **quat 修正可以同步做，但不是前置依赖**：pos 和 quat 可以在同一次 XML 更新里一并修改；但如果只改 pos 不改 quat，当前 pipeline 的抓取结果不会因此变差
3. **预期一步收敛**：jaw midpoint 在 gripper 局部系中只取决于 gripper joint angle，不受 arm 构型影响，因此 v4 pos 写入 XML 后再跑一轮 `11_grasp_center_v4.py`，`residual_local_delta` 应接近 `[0, 0, 0]`
4. **现有 `8_calibrate_grasp_center.py` 仍然可用**：它做的是 position-only 标定，和 `v4` 的 pos 修正目标完全兼容；区别只是 `v4` 用 jaw box 几何直接估计残差，而 `8_calibrate_grasp_center.py` 用 coarse+refine offset sweep 间接搜索


### v3 vs v4 标定 diff 分析

#### 数值对比

| 分量 | default | v3 | v4 suggested | v3→v4 diff |
|------|---------|------|------|------|
| x | -0.0079 | +0.0093 | +0.0139 | **+0.0046** |
| y | -0.0002 | -0.0065 | +0.0024 | **+0.0089** |
| z | -0.0981 | -0.0942 | -0.0821 | **+0.0121** |

三个分量都是正向修正，总位移量 ≈ 12mm。

#### diff 来源

| | v3（`8_calibrate_grasp_center.py`） | v4（`11_grasp_center_v4.py`） |
|--|--|--|
| 起点 | default `[-0.0079, -0.0002, -0.0981]` | v3 `[0.0093, -0.0065, -0.0942]` |
| 参考对象 | cube 在 **gripper body origin** 局部系的 XY 偏移 | **grasp_center** 到 **jaw box midpoint** 的距离 |
| offset_z | `0.0` | `-0.010`（对齐 C60/7.6 baseline） |
| 方法 | world offset sweep → `inv_transform_by_quat` 反推 | jaw 几何中点 → `gripper_world_rot.T` 直接转 |

三项差异叠加产生了 ~12mm 的 diff。其中 z 分量 `+0.0121` 主要来自 offset_z 不同导致的 approach 构型差异；x/y 分量主要来自参考对象不同（gripper origin vs jaw midpoint）。

v1 → v2 → v3 每轮 delta 不同的原因是方法/评分/offset_z 的差异，不是循环依赖。v4 的 jaw-midpoint 方法直接在 gripper 局部系测量，`residual_local = jaw_mid_local(gripper_angle) - gc_local_pos`，与 arm 构型无关，因此预期一步收敛。

