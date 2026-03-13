
## 实验反思

基于 [collison.md](./collision.md) 所有实验的反思:

1. 5-DOF 运动学的硬约束

SO-101 有 5 个有效自由度（shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll），第 6 个是夹爪开合。在一个 top-down pinch grasp 任务中需要同时满足:

约束	DOF 需求	说明
XYZ 位置	3	把 jaw midpoint 送到 cube center
approach 方向 (pitch)	1	夹爪朝下
jaw plane 对齐 (yaw)	1	两爪平面与 cube 侧面平行
总计	5	刚好用完所有 DOF


这意味着**零自由度冗余** -- IK 只有唯一解。


2. 可达下限 vs cube center_z

3. "牵一发而动全身"的本质

每一次调参（offset_xy, approach_z, quat_mode, roll_tuning, rot_mask, cube_height）都是在一个零冗余系统里做扰动。没有冗余意味着：

  * 改 offset → IK 解跳到完全不同的构型
  * 改 cube 高度 → 可达区域边界完全变化
  * 加姿态约束 → 直接变成 over-constrained

**这不是"没搜到好参数"，而是搜索空间的结构本身不支持稳定解**


## 重新设计

**方案A:确定可行工作空间，再配置 cube**

脚本: `35_workspace_mapper.py`（v4：grasp_z 从 `cube_size/2` 到 `cube_size` 以 0.002 步长扫描）

```bash
python 35_workspace_mapper.py --exp-id ws_v4_zsweep_go25 --save /output --gripper-open 25
```

### v4 z-sweep 实测结果（4090, `genesis_poc:latest`, 2026-03-12~13）

grid: X=19 (`0.08~0.26`), Y=21 (`-0.10~0.10`), Z=8 层 (`0.015~0.029`), cube_size=0.03

#### gripper_open 对比（25 / 40 / 50 度）

| grasp_z | open=25 reach | level | feasible | open=40 reach | level | feasible | open=50 reach | level | feasible |
|---------|------|-------|----------|------|-------|----------|------|-------|----------|
| 0.015 | 3 | 66 | 2 | 5 | 34 | 0 | 2 | 19 | 0 |
| 0.017 | 15 | 65 | 10 | 10 | 34 | 4 | 6 | 17 | 3 |
| 0.019 | 53 | 66 | 25 | 22 | 33 | 12 | 13 | 18 | 7 |
| 0.021 | 94 | 65 | 41 | 51 | 29 | 18 | 25 | 18 | 10 |
| 0.023 | 119 | 66 | 50 | 91 | 29 | 22 | 55 | 18 | 14 |
| 0.025 | 142 | 69 | 53 | 119 | 32 | 22 | 86 | 23 | 15 |
| **0.027** | **165** | **66** | **55** | 136 | 32 | 23 | 109 | 25 | 16 |
| 0.029 | 180 | 68 | 55 | 150 | 29 | 21 | 128 | 25 | 16 |
| **jaw_gap** | | **~33.6mm** | | | **~48.1mm** | | | **~57.4mm** | |
| **总 feasible** | | | **291** | | | **56** | | | **44** |

**TODO: 验证下open=30**

#### 关键发现

1. **gripper_open 越大，jaw_gap 越宽但 jaws_level 越差**：open=25 level~66, open=40 level~32, open=50 level~22
2. **对 30mm cube，gap 余量不是瓶颈**。open=25 时 gap=33.6mm > 30mm 已足够，再开大反而因 jaws_level 劣化导致 feasible 减少
3. **open=25 是 cube_size=30mm 的最优 gripper 开角**：feasible=291（总），远超 open=40 (56) 和 open=50 (44)
4. 社区参考：RoboSim SO-101 数据集用 0-100% 百分比控制夹爪（0%=闭合，100%=全开），cube half-size 1.5~2.5cm（全尺寸 3~5cm），cube X=12~24cm Y=-8~8cm

### 33_grasp_light.py 默认参数分析与推荐

**推荐新默认**：
```
cube_x=0.15, cube_y=-0.06, cube_size=0.03
grasp_offset_z=0 
approach_z=0.012 
gripper_open=25
→ pre_close_target_z = 0.015 + 0 + 0.012 = 0.027 (最优 z 层)
```

该点 pos_err=0.27mm, delta_z=-1.46mm, jaw_gap=33.7mm > 30mm — **fully feasible**。









**方案 B：改变抓取策略**
如果 workspace mapping 发现 top-down 抓取的 feasible zone 太小或太边缘：

斜角抓取（angled grasp）：不要求纯 top-down，而是让夹爪以 15-30 度倾斜角接近。这释放了一个姿态自由度，让 5-DOF 系统有 1 DOF 冗余
侧面抓取（side grasp）：对于小 cube，从侧面水平抓取可能更自然
scoop grasp：先把夹爪一侧伸到 cube 下方，再合拢

**方案 C：简化轨迹规划**

当前 33_grasp_light.py 的轨迹规划已经非常复杂：descent waypoints + quat gating + roll recovery + replan + pregrasp clearance。这个复杂度本身就是"在不可行空间里挣扎"的症状。如果方案 A 找到了可行 zone，轨迹应该极其简单





## 基于方案A 推荐 cube 参数的实验 (03-13)


基于新的 cub pos :  `cube_x=0.15, cube_y=-0.06, cube_size=0.03`

执行 33_grasp_light.py 大概:

![image](./images/light_wrist_roll_new_position/f052_approach.png)

是**对角线夹取**。即如果不约束 `wrist-roll`，ik solver + quant=None，最终就会有这个不确定性。

相比，调大gripper_open，接下来建议加 `wrist_roll` 约束

### light wrist roll 实验

**v2 思路：approach 过程中自适应触发 snap**

不硬写"approach 结束后 snap"，而是在 approach 每步检查 jaw midpoint z 与 cube_top_z 的距离：

```
buffer_z = jaw_mid_z - cube_top_z
```

当 `buffer_z` 降到一个阈值时（比如 `snap_buffer = N * delta_z_per_step`，N=3~5 步的缓冲），开始执行 yaw snap：

1. **触发条件**：`buffer_z < snap_trigger`（snap_trigger 由相邻 waypoint 的 z 步长 × 缓冲步数自动计算，不是硬编码常数）
2. **snap 执行**：读取当前 closing_axis yaw → 算最近 0/90° 对齐角 → 逐步插值调整 wrist_roll（分 N 步完成，不是一步跳变）
3. **snap 之后继续 descent**：剩余 waypoint 保持 snapped 的 wrist_roll 作为 seed

这样 snap 在"距 cube 还有几步"时就开始，有足够空间完成旋转，不会在 cube 上方/上面才开始转。

关键参数：
- `snap_trigger`：自适应 = `descent_z_step * snap_buffer_steps`
- `snap_buffer_steps`：3~5（控制提前量）
- snap 插值步数：与 buffer_steps 相同（平滑过渡）

脚本：`37_grasp_yaw_snap.py`



















## 关键指标说明


### cube_size, cube_center_z

```py
# genesis/morphs.py 
def __init__(self, **data):
    super().__init__(**data)

    if self.lower is None or self.upper is None:
        if self.pos is None or self.size is None:
            gs.raise_exception("Either [`pos` and `size`] or [`lower` and `upper`] should be specified.")

        self.lower = tuple((np.array(self.pos) - 0.5 * np.array(self.size)).tolist())
        self.upper = tuple((np.array(self.pos) + 0.5 * np.array(self.size)).tolist())
```

size 是 full dimensions（全尺寸），pos 是 中心。lower = pos - size/2，upper = pos + size/2。

cube_size=0.03 -> 半高=0.015，即 cube_z=0.015(贴地)；

~~当前 cube_z=0.04：cube 初始悬空，settle 后坠落到 z≈0.015。但所有 IK 目标（mid_grasp、pre_close_target）仍然基于 cube_init[2]=0.04 计算，和实际 cube 位置差了 ~2.5cm。这是一个 bug~~

