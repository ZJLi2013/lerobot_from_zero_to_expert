# SO-101 Genesis SDG Best Practices

> 面向可复用的合成数据流程：稳定朝向、可达抓取、可解释调参、可量化验收。

---

## 1) 推荐基线配置

### 1.1 模型与场景

- **MJCF（推荐）**：`so101_new_calib.xml`，来自 HuggingFace `Genesis-Intelligence/assets`
  - 脚本自动通过 `huggingface_hub.snapshot_download` 下载
  - 也可手动指定 `--xml /path/to/so101_new_calib.xml`
- Genesis 加载：`gs.morphs.MJCF(file=..., pos=(0,0,0))`，无需 `fixed=True` 或 `base_offset`

### 1.2 末端执行器

- EE link 搜索顺序：`gripper_link` → `gripperframe` → `gripper` → `Fixed_Jaw`
- MJCF 模型通常匹配到 `gripper`

### 1.3 IK 姿态与 Home

```
IK_QUAT_DOWN = [0.0, 1.0, 0.0, 0.0]   # 夹爪朝下（wxyz）
HOME_DEG     = [0, -30, 90, -60, 0, 0]  # 肩→夹爪，单位°
```

- Home 跟踪误差：MJCF 0.12°（URDF 8.27°，改善 69x）
- 无需 probe quaternion search，`[0,1,0,0]` 已验证可用

### 1.4 相机与工作区

- 顶视 + 侧视同时保留，确保能观察夹爪-目标相对位置和抓取过程
- 目标物初始采样建议在前方舒适区：
  - `x ∈ [0.12, 0.20]`，`y ∈ [-0.05, 0.05]`
  - 避免贴近基座

### 1.5 控制参数

```
KP = [500, 500, 400, 400, 300, 200]
KV = [ 50,  50,  40,  40,  30,  20]
```

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

`approach_z` 是 **IK 目标点**（`gripper` link）相对于方块中心的 z 偏移量。
但 IK 目标点 ≠ 指尖位置，指尖在 gripper frame 下方还有几毫米偏移。

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
2. **offset_z**（进一步微调高度）
3. **offset_x / offset_y**（水平对齐）
4. **gripper_close 角度**（夹持力）
5. **close_hold_steps**（稳定段）
6. （最后）控制增益和速度

---

## 4) Trial 与 Full Episode 的差异

`3_grasp_experiment.py` 的自动调参分两步：先跑快速 trial 找最佳 offset，再跑完整 episode 正式采集。
但两者的轨迹节奏不同，可能导致 **trial 成功但 full episode 失败**：

```
   Trial（快速试探，~72 步）            Full Episode（完整采集，~240 步）

    ┌── approach (15步) ──┐            ┌── approach (30步) ──────┐
    │   快速下降到位        │            │   缓慢下降到位           │
    ├── close (10步) ──┤            ├── close (20步) ──────┤
    │   迅速闭合 → 方块被    │            │   慢慢闭合 → 还没夹紧时    │
    │   夹指卡住              │            │   方块已经滑落              │
    ├── lift (20步) ──┤            ├── close_hold (12步) ─┤
    │   立即提起              │            │   保持但没夹住              │
    └── Δz = 0.011 ✓ ──┘            ├── lift (30步) ────┤
                                       │   空手提起                   │
                                       └── Δz = 0.000 ✗ ──────┘

   快节奏 = 惯性帮忙                    慢节奏 = 纯靠摩擦力
   夹爪还没松方块就被提走                摩擦力不够 → 方块留在原地
```

**解决方案**：

- 增大 `close_hold_steps`（20→30），让夹爪有更多时间建立夹持力
- 降低 `approach_z`，让指尖真正包住方块
- 增大 `gripper_close` 角度，产生更大法向力

---

## 5) 自动调参工具

`3_grasp_experiment.py`（MJCF 版）支持三维 offset 网格搜索：

```bash
python 3_grasp_experiment.py \
  --exp-id E5_deep_approach \
  --auto-tune-offset \
  --approach-z 0.0 \
  --gripper-close 45 \
  --close-hold-steps 25 \
  --offset-x-candidates=-0.01,-0.005,0.0,0.005,0.01 \
  --offset-y-candidates=-0.01,-0.005,0.0,0.005,0.01 \
  --offset-z-candidates=-0.02,-0.015,-0.01,-0.005,0.0
```

自动化逻辑：

1. 固定同一初始 box 位姿
2. 遍历 `(ox, oy, oz)` 候选（grid search）
3. 每个候选跑短 trial（~72 步），计算 `lift_delta`
4. 选择 `lift_delta` 最大的 offset
5. 用最优 offset 跑正式 episode，输出 `.rrd + metrics.json`

> E4 实验教训：100 组搜索中，oz=-0.01 层有 7 组有效（Δz>0.005），
> 而 oz=+0.01 层全部为 0。**approach 高度是成败关键。**

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
| box 完全不动 | approach 太高，指尖没碰到方块 | 降低 `approach_z` (0.02→0.0) |
| Δz 在 0.002~0.01 之间 | 指尖擦到方块但夹不住 | 增大 `gripper_close`，增大 `close_hold` |
| trial 成功但 episode 失败 | 慢节奏下摩擦力不够 | 降低 approach + 增大 close + 增大 hold |
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

## 9) 一句话流程

用 MJCF 官方模型 + gripper-down IK，先降 `approach_z` 保证指尖到方块侧面，再用 `offset → close → hold` 三步法 grid search 迭代，以 `cube_lift_delta > 0.01m` 作为统一验收标准。

---
