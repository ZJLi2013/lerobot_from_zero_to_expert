# SO-101 Genesis SDG Best Practices

> 面向可复用的合成数据流程：稳定朝向、可达抓取、可解释调参、可量化验收。

EE = End Effector（末端执行器）, 就是机械臂最末端、真正和外界接触的那个部件. 对 SO-101 来说，就是夹爪
TCP = Tool Center Point（工具中心点）,是 EE 上的一个具体的参考点, 这里就是 **grasp_center**
通俗说，EE 是"手"，TCP 是"手上真正要对准东西的那个点"

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
- `approach_z`、`offset_x / offset_y / offset_z` 不是“绝对世界坐标”，而是**相对于方块中心的偏移量**
  - 脚本里的 approach 目标可近似理解为：
  - `target_pos = cube_center_world + [offset_x, offset_y, offset_z] + [0, 0, approach_z]`
- 因此：
  - `approach_z=0.012` 的意思不是“世界坐标 z=0.012m”
  - 而是“抓取目标点位于方块中心上方 `0.012m`（12mm）”
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

- `grasp_center` 可以理解为“真正希望对准目标物中心的那个参考帧”
- 对夹爪类末端来说，它通常不是：
  - 腕部 link 原点
  - 单侧手指/jaw 的 link 原点
- 它更接近：
  - 两个夹指之间的 pinch center
  - 或者工具中心点（TCP, tool center point）

Genesis 最新源码体现出的官方思路非常明确：

1. **要么资产里已经有合适的 TCP frame**
   - 例如 Franka 的 MJCF 直接放了 `end_effector` site
   - 也就是说，资产作者先把“末端该对哪里”定义清楚
2. **要么 IK API 支持对 link 上的偏移点求解**
   - 即 `local_point`
   - 让“link 原点之外的 TCP 点”去对齐目标

对当前 SO-101 的启发是：

- 现在继续做 `offset_x / offset_y / offset_z` 试错，本质上是在补偿“抓取参考点没定义对”
- 这种补偿在少数 trial 上可能有效，但它会：
  - 对 cube 位置敏感
  - 对姿态变化敏感
  - 对不同 IK 解敏感
- 所以 `offset` 更适合作为 **微调项**，而不是用来替代 `grasp_center`

一句话说：

> `offset` 应该是在“已经有正确 grasp_center/TCP”的前提下做毫米级微调，
> 而不是在“没有 grasp_center”的情况下去硬补几厘米级结构误差。

对当前 SO-101 官方 MJCF，一个很关键的观察是：

- 它已经包含一个 `gripperframe` site
- 这说明资产作者其实已经给出了一个“夹爪参考帧”的候选
- 但 `Genesis 0.4.0` 运行时不能像新版本那样直接对 site/local_point 做 IK

因此，**最推荐的兼容旧版 Genesis 方案**不是继续做 proxy，而是：

1. 保留官方 `so101_new_calib.xml`
2. 在 `gripperframe` 的同位置、同姿态上增加一个固定 `body/link`
3. 将这个固定 link 命名为 `grasp_center` 或 `ee_tcp`
4. 后续 IK 直接对这个 link 求解

这相当于把“最新 Genesis 里的 `local_point/TCP` 语义”提前固化进资产本身。

---

## 2) Pick-Place 轨迹结构

抓取采集的核心是一条 **scripted 轨迹**（非学习产生），机械臂按 6 个阶段顺序执行：

```
                    ┌─ [1] move_pre ─────────┐
                    │  从 Home 移到方块上方    │
                    ▼                        │
              ┌─── (pre_grasp) ◉             │
              │     z = cube + 0.10m         │   [6] return
              │                              │   回到 Home
   [2] approach│                              │
              │                              │
              ▼                              │
         (approach) ◉  ← 这里是关键！        │
              │         z = cube + approach_z │
   [3] close  │  夹爪从 open → close          │
              ▼                              │
         (close_hold) ◉  保持夹紧             │
              │                              │
   [4] lift   │  带着方块上升                  │
              ▼                              │
           (lift) ◉                          │
              │    z = cube + 0.15m          │
              └──────────────────────────────┘
```

每个阶段之间用 **线性插值（lerp）** 过渡，步数由 `episode_length × fps` 按比例分配。

---

## 3) 调参核心原理

### 3.1 approach_z —— 最关键的参数

`approach_z` 是 **抓取参考点** 相对于方块中心的 z 偏移量。直白解释，抓手往下落多少开始夹。太高，抓手只能碰到box 上表面，包不住；太低，估计把box 挤走了。

这个“抓取参考点”必须先定义对，才能谈 `approach_z` 是否合适。

> 最新实验（E9–E12）表明：`gripper` 不能直接当作 SO-101 的抓取参考点；
> 当前更接近 jaw 的是 `moving_jaw_so101_v1`，但它仍然不是理想的“夹爪中心”。

```
        approach_z = 0.02（默认，太高）       approach_z = 0.0（更低，接近可抓）

     z(m)                                     z(m)
     0.05 ┤                                   0.05 ┤
          │                                        │
     0.04 ┤                                   0.04 ┤
          │    ┌───┐ ← IK 目标点                    │
     0.035┤    │ ◉ │   (gripper frame)              │
          │    │   │                           0.03 ┤ ┌─────┐ ← 方块顶部
     0.03 ┤ ┌──┤   ├──┐ ← 方块顶部                  │ │     │
          │ │  └───┘  │                             │ │     │
     0.015┤ │ ██████ │ ← 方块中心              0.015┤ │██◉██│ ← IK 目标 = 方块中心
          │ │ ██████ │                              │ │█████│   指尖在两侧包住方块
     0.00 ┤ └────────┘ ── 桌面                 0.00 ┤ └─────┘ ── 桌面
          └──────────────────                      └──────────────────

    问题：指尖在方块顶部上方                    正确：指尖能包住方块两侧
    夹爪"虚握"→ 方块不动                       夹爪产生接触力 → 可以抬起
```

### 3.2 offset (ox, oy, oz) —— 微调对齐

IK 求解出的夹爪位置与方块中心之间可能存在系统偏差。
`offset` 是施加在 approach/close/lift 阶段的**额外空间微调**：

> 重要：`offset` 只能解决“小偏差”。如果当前用的是错误的 EE / TCP 参考点，
> 那么 `offset` 会退化成“拿大范围试错去补资产建模问题”，可复用性会很差。

```
            俯视图（从上往下看）
      y
      ↑
      │    每个 · 是一组 (ox, oy) 候选
      │
 +0.01│  ·    ·    ·    ·    ·
      │
+0.005│  ·    ·    ·    ·    ·
      │
  0.0 │  ·    ·    ◆    ·    ·     ◆ = 方块中心（IK 默认目标）
      │                ↑
-0.005│  ·    ·    ·   ★ ·    ·     ★ = 找到的最佳 offset
      │
 -0.01│  ·    ·    ·    ·    ·
      └──────────────────────→ x
        -0.01      0.0     +0.01
```

侧视图（oz 维度效果）：

```
     oz > 0（太高）          oz = 0（默认）           oz < 0（更低）
        ┌─┐                    ┌─┐                    ┌─┐
        │ │ ← 指尖             │ │                    │ │
        └─┘                    └─┘                    └─┘
                            ┌──────┐               ┌──────┐
      ┌──────┐              │██████│               │██│ │██│
      │██████│              │██████│               │██└─┘██│ ← 指尖在方块侧面
      └──────┘              └──────┘               └──────┘

     完全不接触             偶尔擦到                 有接触，可能抓住
     Δz ≈ 0               Δz ≈ 0.005              Δz ≈ 0.01+
```

### 3.3 gripper close 角度

```
     close = 25°（不够）              close = 45°（推荐尝试）

        ┌─┐       ┌─┐                  ┌┐     ┌┐
        │ │       │ │                  ││     ││
        │ │       │ │                  ││ ███ ││ ← 方块被夹住
        │ │  ███  │ │ ← 间隙太大       ││ ███ ││
        │ │  ███  │ │   方块可滑落      └┘     └┘
        └─┘       └─┘
```

### 3.4 先几何、后动力学

调参顺序建议（从最高优先级到最低）：

1. **approach_z**（让指尖到方块侧面高度）
   - 通俗说，就是“手先落到多高的位置再开始夹”；太高会只碰到上面，太低又可能顶桌子或把方块挤跑。
2. **offset_z**（在 `approach_z` 基础上继续微调高度）
   - 通俗说，就是“在当前高度上再往下压一点点，或者再抬一点点”。
3. **offset_x / offset_y**（水平对齐）
   - 通俗说，就是“手已经大致到位后，再往前后左右挪一点，让方块更居中地进入两爪之间”。
4. **gripper_close 角度**（夹持力）
   - 通俗说，就是“手指最后合到多紧”；不是越紧越好，太猛也可能把方块挤走。
5. **close_hold_steps**（稳定段）
   - 通俗说，就是“手指合上之后，不是立即抬，而是先稳稳地捏住一小会儿”。
6. （最后）控制增益和速度
   - 通俗说，就是“前面的几何位置都差不多对了，才去调动作快慢、软硬和跟随手感”。

---

## 4) Trial 与 Full Episode 的差异

`3_grasp_experiment.py` 会先跑短 trial 做 offset 搜索，再跑 full episode 做正式验证。

当前更准确的理解是：

- trial 适合快速筛掉明显无效的 offset
- full episode 才能判断“是否真的形成稳定夹持”
- **trial 变好不等于根因已解决**
- 在本项目里，早期的 trial/full-episode gap 一度误导了分析；后续确认，真正的结构性根因首先是 **EE/TCP 定义错误**

因此最佳实践是：

1. 先确保 `grasp_center/TCP` 定义正确
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
   - 这说明“EE/TCP 定义错误”这个主冲突已经解决。



### 9.2 关键里程碑

| 实验 | 当前应记住的结论 |
|------|------------------|
| E20-E24 | 进入真正抓取调参阶段；`approach_z=0.012` 附近优于 `0.01/0.015`，`hold` 增加有帮助，但“更大闭合角”不一定更好 |
| E30 | 双侧斜视角 + 中间稳区采样 + auto-tune-offset | 跟新camera pose 方便debug |
| E36 - 37 | 独立 gripper 标定：扫 `-20,-10,0,10,20,30,40,50` 并导出 PNG | 已确认 **gripper 数值越大，夹爪张口越大**；之前把正值当“close”在物理上是错误的 |
| E38-E41 | 固定 `E33` 的 pose/几何后，测试 `gripper_close=-20,-10,0` | 三者 episode 都约 `+0.0006m`；gripper_close 方向问题已经从“未知”变成“已知”，但抓取失败的主因仍然是 TCP 目标点没有把 box 放进 pinch 区域。|



### 9.4 当前建议

固定 gripper_close=-20 或另一个已经确认“确实在闭合”的值
固定 approach_z=0.012
固定 offset_z=-0.010
回到固定 pose 的中心附近 x/y 小范围扫描
不要再让 auto-tune 只看 delta_z,要把“是否真的进入两爪中间”作为第一筛选条件 **哪个 x/y 让 box 在 close 开始前就处在两爪之间**










---
