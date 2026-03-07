# SO-101 MJCF XML 版本说明

## 版本清单

| 文件 | 来源 | `grasp_center` pos（gripper 局部坐标） | `frictionloss` | 备注 |
|------|------|--------------------------------------|----------------|------|
| `so101_new_calib.xml` | v1 标定脚本输出 | `[-0.0297, -0.0039, -0.0957]` | 默认 `0.1/0.052` | v1 标定过冲，偏到了 box 另一侧 |
| `so101_new_calib_default.xml` | HuggingFace 官方原版 | `[-0.0079, -0.0002, -0.0981]` | 默认 `0.1/0.052` | 官方 TCP，jaw 偏右偏后 |
| `so101_new_calib_v1_half.xml` | v1 标定量 / 2 | `[-0.0188, -0.0021, -0.0969]` | 默认 `0.1/0.052` | 折中补偿，xy 接近 box 边缘 |
| `so101_new_calib_v2.xml` | v2 标定脚本（优先 xy 居中） | `[0.0004, 0.0069, -0.1023]` | 默认 `0.1/0.052` | xy 改善，但 z 方向 IK 偏高 |
| `so101_new_calib_v3.xml` | v3 标定脚本（加 IK sanity） | `[0.0093, -0.0065, -0.0942]` | 默认 `0.1/0.052` | 当前主线。E42-E49、C54-C56 的基线 |
| `so101_new_calib_v3_nofrictionloss.xml` | v3 基础上 `frictionloss` 置零 | 同 v3 | 全部 `0` | collision 排查用，C55 实验 |

## 标定版本演进

```
官方 default
    ↓  v1 标定（lift_delta 优先）→ 过冲
    ↓  v1_half（取一半）→ 折中
    ↓  v2 标定（xy 居中优先）→ z 偏高
    ↓  v3 标定（加 IK sanity 约束）→ 当前主线
```

详细标定逻辑和数值见 `02_intermediate/doc/calib.md`。

## 各版本区别汇总

所有版本的区别只在两处：
1. `<body name="grasp_center" pos="...">` — 夹持 TCP 的局部坐标
2. `frictionloss` — 仅 `_nofrictionloss` 版本把关节摩擦置零

其余（mesh、link 结构、joint 范围、actuator 等）完全一致，都基于同一份 onshape-to-robot 导出的 SO-101 模型。

## `assets/assets/` 子目录

包含 mesh STL 文件和早期辅助 XML，供 MJCF 加载器引用：
- `*.stl` — SO-101 各 link 的 mesh
- `scene.xml` / `joints_properties.xml` — 早期调试用，非当前主线

