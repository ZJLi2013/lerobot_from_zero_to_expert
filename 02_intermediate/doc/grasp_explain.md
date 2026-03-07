## 1) 关键概念术语表

### 1.1 SO-101 夹爪相关概念

```
SO-101 gripper 结构示意（侧视，close 状态）

     ┌──────────────┐
     │  gripper body │  ← "gripper" link = 腕部/夹爪基座
     │  (body origin)│     它的坐标原点在这里，不在指尖
     └──────┬───────┘
            │
    ┌───────┴────────┐
    │                │
    ▼                ▼
 ┌─────┐        ┌─────┐
 │fixed│        │moving│  ← moving_jaw_so101_v1 (活动爪)
 │ jaw │  ████  │ jaw  │
 │     │  ████  │     │     ████ = 被夹住的 cube
 └──┬──┘        └──┬──┘
    │              │
    └──────◉───────┘
           ↑
      jaw pinch center（夹持中心点）
      = 两指尖闭合时的中心接触点
```

| 概念 | 英文 | 说明 |
|------|------|------|
| **EE** | End Effector | 末端执行器，对 SO-101 来说就是整个夹爪机构 |
| **TCP** | Tool Center Point | 工具中心点，EE 上的一个参考点，IK 的目标。在本项目中 = `grasp_center` body |
| **gripper** | — | MJCF 中的 link，对应腕部/夹爪基座。注意：它的 body origin 在基座，不在指尖 |
| **fixed jaw** | — | 固定爪（不动的那一侧），是 gripper body 几何的一部分 |
| **moving jaw** | moving_jaw_so101_v1 | 活动爪（由 `gripper` joint 驱动的那一侧） |
| **jaw pinch center** | 夹持中心点 | fixed jaw 和 moving jaw 闭合时，两个指尖之间的中心接触点。**这是物体被夹住的真正位置** |
| **grasp_center** | — | MJCF 中人为添加的固定 body，挂载在 gripper body 下。设计意图是作为 TCP 让 IK 能直接求解到夹持中心点 |
| **gripperframe** | — | 官方 MJCF 中的 site，`grasp_center` body 复制了它的 pos/quat |
| **body origin** | link 坐标原点 | 每个 link body 的坐标原点。注意：body origin ≠ mesh 的视觉最远端（指尖） |

### 1.2 grasp_center 与 jaw pinch center 的关系

```
理想情况：grasp_center = jaw pinch center

     IK 目标 → grasp_center ──精确对准──→ jaw pinch center ──对准──→ cube center
                 (0.2mm 误差)              (两者重合)                  (cube 在两爪中间)

     ✓ IK 把 grasp_center 送到 cube → 夹持中心点也在 cube → 两爪包住 cube → 抓取成功
```

### 1.3 一次完整抓取的全过程（从 MJCF 定义到夹爪闭合）

下面用**实际流程顺序**把所有概念串起来。

#### 第 1 步：MJCF 里"安装"grasp_center

在写 MJCF XML 的时候，需要在 gripper body 上标记一个参考点，告诉 IK 系统"这就是我希望对准目标物的那个点"。这个参考点就是 `grasp_center` body。

关键：`grasp_center` 是**写在 XML 文件里的一行配置**，可以随时修改。它不是物理零件，是一个**虚拟标记**。
这里的 `pos="-0.008 -0.0002 -0.098"` 是**相对于 `gripper` body 的局部坐标系**来写的。

```xml
<!-- 在 gripper body 内部 -->
<body name="grasp_center" pos="-0.008 -0.0002 -0.098" quat="0.707 0 0.707 0">
```

这行配置的意思是：在 `gripper` body 的局部坐标系中，往 `(-0.008, -0.0002, -0.098)` 方向偏移，放一个标记。

与此同时，手臂上还有一个**物理上不可改变**的点：jaw pinch center（夹持中心点）——两个 jaw 指尖闭合时的接触中心。这个点由 jaw 的几何形状决定，改不了。
从坐标角度看，它也可以理解成“**相对于 `gripper` 局部坐标系固定的一个点**”；只是它当前没有被单独写成 XML 里的 body/site，而是由 jaw 的几何关系隐含决定。

```
gripper body 上的两个点：

  ┌──────────── gripper body ────────────┐
  │                                      │
  │    ◆ grasp_center                    │
  │    (虚拟标记，位置由 XML 定义，可改)    │
  │                                      │
  │              ★ jaw pinch center       │
  │              (物理接触点，由 jaw 几何    │
  │               形状决定，改不了)         │
  └──────────────────────────────────────┘

  ◆ 和 ★ 之间有距离 → 这就是 bug 的来源
```

#### 1.2.1 用坐标变换来理解“为什么两个都焊死，但还是会错位”

这里最容易困惑的问题是：

> `grasp_center` 也是焊死在 gripper 上的，`jaw pinch center` 也是焊死在 gripper 上的，
> 那它们怎么还会不重合？

关键是：**它们都固定在同一个 `gripper` 上，但固定的是两个不同的局部坐标点。**

先都写成 `gripper` 局部坐标系下的点：

```text
p_grasp_local = grasp_center 在 gripper local frame 下的位置
p_pinch_local = jaw pinch center 在 gripper local frame 下的位置
```

当前 bug 的本质就是：

```text
p_grasp_local != p_pinch_local
```

当机械臂运动时，`gripper` 整体发生刚体变换。设 `gripper` 在世界坐标系中的位姿是：

```text
R_gripper, t_gripper
```

那么这两个点在世界坐标系中的位置分别是：

```text
p_grasp_world = R_gripper * p_grasp_local + t_gripper
p_pinch_world = R_gripper * p_pinch_local + t_gripper
```

更直观地说：

- `grasp_center` 是你在 gripper 上**人为贴的红点**
- `jaw pinch center` 是夹爪真正**咬合的物理点**
- 问题不是红点会乱跑，而是红点从一开始就贴错了位置

修复:: **直接修改 XML，把 `grasp_center` 在 gripper local frame 下的定义位置，改成 jaw pinch center 的那个局部位置**。

#### 第 2 步：IK 求解

脚本运行时，IK 系统的输入是：
- **目标位置**：cube_center，比如 `[0.16, 0, 0.015]`
- **要对准目标的那个点**：`grasp_center`（XML 中定义的那个虚拟标记）

IK 的任务是：算出一组关节角度，让整条手臂运动后，`grasp_center` 这个标记正好落在 cube_center 的位置。

IK 做得很好——精度 0.2mm。`grasp_center` 确实到了 cube_center。

#### 第 3 步：手臂运动，所有点跟着一起动

手臂按照 IK 算出的关节角度运动。`grasp_center` 到达了 cube_center——这没问题。

但 jaw pinch center 也焊在同一条手臂上，它跟着一起运动了。因为 ◆ 和 ★ 在手臂上的位置不同，当 ◆ 到达 cube_center 时，★ 到达的是 cube_center 旁边的某个位置。

```
运动后的世界坐标：

          cube_center [0.16, 0, 0.015]
               │
               ▼
       ──── ◆ ────  grasp_center 精确到达 ✓
               │
               │  ← 这段距离就是 ◆ 和 ★ 在手臂上的间距（投影到世界坐标后）
               │
       ──── ★ ────  jaw pinch center 到了这里
               │
               ▼
          cube 在 jaw 外缘，不在两爪中间 ✗
```

#### 第 4 步：close — 夹爪闭合

夹爪开始闭合。但因为 jaw pinch center（两指尖的夹持点）偏离了 cube，cube 在 jaw 外缘而不是中间。闭合时只是擦边推动 cube → **空抓**。

这就是 E41/E42/E44 的 debug PNG 中看到的画面。

#### 修复：在 XML 中把 ◆ 挪到 ★ 的位置

`jaw pinch center`（★）是物理决定的，改不了。但 `grasp_center`（◆）只是一行 XML 配置，可以改。

```
修复前（当前 XML）：
  gripper body 上：  ◆ 在位置 A，★ 在位置 B     → A ≠ B → 空抓

修复后（改 XML）：
  gripper body 上：  ◆ 移到位置 B，★ 在位置 B   → A = B → IK 对准 ◆ = 对准 ★ = 对准 cube → 能抓住
```

具体操作：修改 MJCF 中 `grasp_center` 的 `pos` 属性（当前 `pos="-0.008, -0.0002, -0.098"`），把它改成 jaw pinch center 在 gripper 局部坐标系中的位置。这个新值需要通过实验标定（例如精细网格搜索 offset_x / offset_y 找到让 cube 正好落在两爪中间的补偿值，反推出正确的 local pos）。

#### 1.3.2 为什么“官方提供的 grasp_center”仍然可能需要标定

一个很自然的问题是：

> `grasp_center` 不是官方在 `so101_new_calib.xml` 里已经给了吗？
> 为什么还会有“标定不准”的问题？

关键要分清两件事：

1. **官方确实给了一个 frame / site**
2. **这个 frame 是否等于你当前抓取任务真正需要的夹持中心点，需要实验验证**

从 XML 可以看到：

```xml
<!-- Frame gripperframe -->
<site name="gripperframe" pos="-0.0079 -0.000218121 -0.0981274" ... />

<!-- Fixed TCP/grasp-center body for runtimes that cannot IK directly to sites/local points -->
<body name="grasp_center" pos="-0.0079 -0.000218121 -0.0981274" ...>
```

这说明：

- `grasp_center` 是从 `gripperframe` 这个官方 frame/site 复制出来的
- 它的主要作用是：**给不能直接 IK 到 site 的 runtime 一个可求解的 body**

也就是说，它首先是一个：

- 建模参考点
- 工具参考 frame
- runtime 兼容点

但它**不一定自动等于**：

- 当前这个“top-down 抓 3cm 方块”任务里，两个 jaw 真正夹住方块时的**jaw pinch center**

换句话说：

> “官方提供了 `grasp_center`” 只说明“官方给了一个参考点”，  
> 不等于 “这个参考点已经被任务级验证为真实夹持中心点”。

这在机器人资产里很常见。一个 frame 可能只是：

- 机械设计参考点
- 工具安装参考点
- 可视化参考点
- 兼容旧版 runtime 的 IK 参考点

这些都不一定是“抓一个小方块时最合适的 TCP”。

所以这里说的“标定”，不是在否定官方资产，而是在做一件更具体的事情：

> 对“SO-101 top-down 抓方块”这个任务，验证当前 `grasp_center` 是否真的等于 jaw pinch center。  
> 如果不等，就把它改到真正的夹持中心位置。

也可以把它理解成：

- **资产作者**负责给出一个可用的 frame
- **任务开发者**负责验证这个 frame 是否适合当前任务

如果任务验证发现：

```text
grasp_center != jaw pinch center
```

那就需要做任务级校准，把它改成：

```text
grasp_center := jaw pinch center
```

### 1.4 body origin vs mesh tip

```
        body origin（坐标原点）
              ↓
     ┌────────◉────────┐
     │    gripper body  │
     │                  │
     │    ┌──────────┐  │
     │    │  motor   │  │
     │    └──────────┘  │
     │         │        │
     │    ┌────┴─────┐  │
     │    │          │  │
     │    │  jaw     │  │
     │    │  mesh    │  │
     │    │          │  │
     │    └────◉─────┘  │   ← mesh tip（指尖）
     └──────────────────┘

     body origin z = 0.104m ← diag_link_positions 输出的值
     mesh tip    z ≈ 0.015m ← 实际指尖到达的高度（从 PNG 确认）
     差距 ≈ 0.089m

     教训：不能用 body origin 的 z 来判断"指尖能不能到达某个高度"
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
