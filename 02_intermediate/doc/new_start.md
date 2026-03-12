
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



**方案A:确定可行工作空间，再配置 cube**

脚本: `35_workspace_mapper.py`（v4：grasp_z 从 `cube_size/2` 到 `cube_size` 以 0.002 步长扫描）

```bash
python 35_workspace_mapper.py --exp-id ws_v4_zsweep_go25 --save /output --gripper-open 25
```

### v4 z-sweep 实测结果（4090, `genesis_poc:latest`, 2026-03-12）

grid: X=19 (`0.08~0.26`), Y=21 (`-0.10~0.10`), Z=8 层 (`0.015~0.029`), 共 3192 pts

| grasp_z | reachable | jaws_level | feasible | feasible% |
|---------|-----------|------------|----------|-----------|
| 0.015 | 3 | 66 | 2 | 0.5% |
| 0.017 | 15 | 65 | 10 | 2.5% |
| 0.019 | 53 | 66 | 25 | 6.3% |
| 0.021 | 94 | 65 | 41 | 10.3% |
| 0.023 | 119 | 66 | 50 | 12.5% |
| 0.025 | 142 | 69 | 53 | 13.3% |
| **0.027** | **165** | **66** | **55** | **13.8%** |
| 0.029 | 180 | 68 | 55 | 13.8% |

**关键发现**: 瓶颈是 reachable（3→180），不是 jaws_level（稳定 65~69）。
z 每升高 2mm，可达性持续改善。最优 z=0.027，feasible zone: X=[0.09,0.21], Y=[-0.1,0.1]。

### 33_grasp_light.py 默认参数可行性分析

当前默认值：
- `cube_z = CUBE_SIZE[2]/2 = 0.015`（贴地）
- `grasp_offset_z = -0.01`
- `approach_z = 0.012`
- `pre_close_target_z = cube_z + offset_z + approach_z = 0.015 + (-0.01) + 0.012 = 0.017`

所以 33 脚本的实际抓取高度对应 mapper 中 **grasp_z = 0.017**。

查 `(0.16, 0.0)` 在各 z 层的数据：

| grasp_z | pos_error_3d | pos_error_z | delta_z | reachable | jaws_level | feasible |
|---------|-------------|-------------|---------|-----------|------------|----------|
| 0.015 | 16.27mm | 15.15mm | 11.19mm | x | x | x |
| **0.017** | **9.10mm** | **8.79mm** | **10.94mm** | **x** | **x** | **x** |
| 0.019 | 6.88mm | 6.87mm | 10.49mm | x | x | x |
| 0.021 | 5.17mm | 4.74mm | 10.19mm | x | x | x |
| 0.023 | 3.58mm | 2.88mm | 10.28mm | **v** | x | x |
| 0.025 | 1.82mm | 1.17mm | 10.56mm | **v** | x | x |
| 0.027 | **0.08mm** | **0.08mm** | 10.76mm | **v** | x | x |
| 0.029 | 0.22mm | 0.17mm | 10.72mm | **v** | x | x |

**结论**：
1. `(0.16, 0.0)` 在 **z=0.017**（当前默认抓取高度）**不可达**（pos_error=9.1mm，阈值 5mm）
2. 即使把 z 提高到 0.023 以上使其 reachable，**delta_z 始终 ~10mm >> 4mm 阈值**——该 XY 位置的 jaws 始终不水平
3. `(0.16, 0.0)` 在所有 z 层都**不是 feasible 点**

**但 (0.16, 0.0) 并非完全无望**——它的问题是 `y=0`（正前方）导致 delta_z 大。mapper 数据中 feasible 点集中在 y 偏侧的位置（如 `(0.13, -0.09)` 或 `(0.17, +0.06)`）。

### 可选改进方向

1. **调高抓取目标**：加大 `approach_z`（如 0.015→目标 z=0.02）或加大 `cube_size`（0.05→center_z=0.025）
2. **偏移 cube_y**：把 cube 放到 y≠0 的位置（如 y=-0.06），让 delta_z 进入可行区
3. **放弃 top-down**：转方案 B（angled/side grasp）

结果文件：`02_intermediate/remote_results/ws_v4_zsweep_go25/workspace_summary.json`



**方案 B：改变抓取策略**
如果 workspace mapping 发现 top-down 抓取的 feasible zone 太小或太边缘：

斜角抓取（angled grasp）：不要求纯 top-down，而是让夹爪以 15-30 度倾斜角接近。这释放了一个姿态自由度，让 5-DOF 系统有 1 DOF 冗余
侧面抓取（side grasp）：对于小 cube，从侧面水平抓取可能更自然
scoop grasp：先把夹爪一侧伸到 cube 下方，再合拢

**方案 C：简化轨迹规划**

当前 33_grasp_light.py 的轨迹规划已经非常复杂：descent waypoints + quat gating + roll recovery + replan + pregrasp clearance。这个复杂度本身就是"在不可行空间里挣扎"的症状。如果方案 A 找到了可行 zone，轨迹应该极其简单





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

