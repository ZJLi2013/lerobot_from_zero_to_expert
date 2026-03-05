# SO-101 Genesis SDG Best Practices

> 面向可复用的合成数据流程：稳定朝向、可达抓取、可解释调参、可量化验收。

关联文档：
- 入口导航：`readme.md`
- 实现流程：`genesis_sdg.md`

---

## 1) 推荐基线配置

### 1.1 URDF 与场景

- URDF：`haixuantao/dora-bambot` 的 `so101.urdf`
- Genesis 加载建议：
  - `fixed=True`
  - `pos=(0.163, 0.168, 0.0)`（将工作区域对齐到场景原点附近）
  - `euler` 从 `(0,0,0)` 起，若朝向异常再做小步调整

### 1.2 末端执行器定义

- 抓取优先使用 `gripperframe`（TCP 语义更清晰）
- 仅在不存在 `gripperframe` 时 fallback 到 `gripper`

### 1.3 相机与工作区

- 顶视与侧视同时保留，确保能观察：
  - 夹爪-目标相对位置
  - 抓取后 box 是否随夹爪抬升
- 目标物初始采样建议在前方舒适区：
  - `x in [0.12, 0.20]`
  - `y in [-0.05, 0.05]`
  - 避免贴近基座区域

### 1.4 控制参数

- 建议起点：
  - `gripper_open=70`
  - `gripper_close=20`
  - `close_hold_steps=12`
- PD 建议高于默认弱阻尼（例如 2-5x），优先保证位置跟踪稳定

---

## 2) 调参核心原理

### 2.1 先几何、后动力学

抓取失败最常见原因是几何误差，不是“力不够”。

调参顺序建议：

1. **TCP/抓取点对齐**（最重要）
2. **平面偏移 `offset_x/offset_y`**
3. **夹爪开合 `open/close`**
4. **闭合稳定段 `hold`**
5. （最后）控制增益和速度细节

### 2.2 为什么优先调 `offset_y`

若现象是“box 在单侧爪外缘”，通常是横向偏差主导。  
先扫 `offset_y` 可最快把 box 拉回两爪中缝；再用 `offset_x` 微调咬合深度。

---

## 3) 标准调参实践（1 episode 快速迭代）

### Step A: 固定控制参数

- 固定：`open=70`, `close=20`, `hold=12`
- 只调抓取偏移：
  - `offset_y ∈ {-0.010, -0.005, 0, +0.005, +0.010}`
  - 在最佳 y 上，再扫：
  - `offset_x ∈ {-0.008, -0.004, 0, +0.004, +0.008}`

### Step B: 固定偏移后调夹爪策略

- `close ∈ {10, 15, 20, 25}`
- `hold ∈ {12, 20, 30}`
- 目标：在不触发异常碰撞的前提下提高 lift 成功率

### Step C: 复验

- 用最佳参数再跑 1 episode，输出 `.rrd + metrics.json`
- 截图记录关键时刻：approach / close / hold / lift

---

## 4) 自动化调参建议

`sdg_so101_grasp_experiment.py` 已支持 offset 网格搜索：

- `--auto-tune-offset`
- `--offset-x-candidates=...`
- `--offset-y-candidates=...`

自动化逻辑：

1. 固定同一初始 box 位姿
2. 遍历 `(offset_x, offset_y)` 候选
3. 每个候选跑短 trial，计算 `lift_delta`
4. 选择 `lift_delta` 最大的 offset
5. 用最优 offset 跑正式 episode

> 建议：offset 自动化后，再加一轮 `close/hold` 网格搜索，通常比单独扫 offset 更容易突破 0 成功率。

---

## 5) 判定逻辑（Metrics & Rules）

### 5.1 基础指标定义

- `cube_z_before_close`：进入 `close` 阶段前 box 的高度
- `cube_z_after_lift`：`lift` 阶段末 box 的高度
- `cube_lift_delta = cube_z_after_lift - cube_z_before_close`

### 5.2 抓取成功判定

推荐阈值：

- `grasp_success = 1` 当 `cube_lift_delta > 0.01 m`
- 否则 `grasp_success = 0`

### 5.3 分级判读（便于调参）

- **成功**：`delta > 0.01 m`
- **临界接触**：`0.002 m < delta <= 0.01 m`（说明已接触但夹持不稳）
- **未抓取**：`delta <= 0.002 m`

### 5.4 辅助检查项

- `state/action` 是否在关节限位内
- close 后是否出现明显抖动或反弹
- side 视图中 box 是否进入两爪中间，而非外侧擦边

---

## 6) 结果记录模板（建议）

每次实验至少保存：

- `improved_sdg_<exp_id>.rrd`
- `metrics_<exp_id>.json`
- 3 张截图：`approach`, `close_end`, `lift_peak`

`metrics.json` 推荐字段：

- `exp_id`
- `ee_link`
- `selected_grasp_offset`
- `gripper_open_deg`, `gripper_close_deg`, `close_hold_steps`
- `cube_z_before_close`, `cube_z_after_lift`, `cube_lift_delta`, `grasp_success`
- `state_range_deg`, `action_range_deg`

---

## 7) 常见失败模式 -> 对应动作

- **box 在单侧爪外缘** -> 先调 `offset_y`
- **box 在爪前方被推走** -> 减小 `offset_x` 或增大 `open`
- **close 后有接触但 lift 掉落** -> 增大 `hold`，再微调 `close`
- **动作幅度异常大** -> 回查 TCP link 与 IK 目标姿态是否一致

---

## 8) 一句话流程

先用正确 TCP 与工作区保证几何可抓，再用 `offset -> close -> hold` 三步法迭代，并用 `cube_lift_delta` / `grasp_success` 做统一验收。

---

## 9) 最小命令清单

以下命令按“本地 -> 4090 -> 回传”最小闭环组织，可直接复用。

### 9.1 本地提交并推送（仅当脚本有更新）

```powershell
cd "c:\Users\zhengjli\Documents\github\lerobot_from_zero_to_expert"
git add "02_intermediate/best_practices.md" "02_intermediate/scripts/sdg_so101_grasp_experiment.py"
git commit -m "update grasp best practices and tuning script"
git push origin main
```

### 9.2 4090 拉取最新代码

```bash
ssh david@<4090_HOST> "cd ~/github/lerobot_from_zero_to_expert && git pull origin main"
```

### 9.3 跑 1 episode（自动 offset 调参）

```bash
ssh david@<4090_HOST> "mkdir -p ~/sdg_grasp_exp && docker run --rm --gpus all \
  -v ~/github/lerobot_from_zero_to_expert:/workspace/lfzte \
  -v ~/sdg_grasp_exp:/output \
  genesis_poc:latest \
  python -u /workspace/lfzte/02_intermediate/scripts/sdg_so101_grasp_experiment.py \
  --exp-id E3_auto_offset \
  --episodes 1 \
  --episode-length 6 \
  --save /output \
  --auto-tune-offset \
  --offset-x-candidates=-0.008,-0.004,0.0,0.004,0.008 \
  --offset-y-candidates=-0.010,-0.005,0.0,0.005,0.010 \
  --gripper-open 70 \
  --gripper-close 20 \
  --close-hold-steps 12"
```

### 9.4 回传结果到本地

```powershell
scp david@<4090_HOST>:~/sdg_grasp_exp/E3_auto_offset/improved_sdg_E3_auto_offset.rrd `
  "c:\Users\zhengjli\Documents\github\lerobot_from_zero_to_expert\02_intermediate\sdg_data\improved_sdg_E3_auto_offset.rrd"

scp david@<4090_HOST>:~/sdg_grasp_exp/E3_auto_offset/metrics.json `
  "c:\Users\zhengjli\Documents\github\lerobot_from_zero_to_expert\02_intermediate\sdg_data\metrics_E3_auto_offset.json"
```

### 9.5 本地快速查看结果

```powershell
rerun "c:\Users\zhengjli\Documents\github\lerobot_from_zero_to_expert\02_intermediate\sdg_data\improved_sdg_E3_auto_offset.rrd"
```

判定只看两个值即可：

- `grasp_success`
- `cube_lift_delta`（目标 > `0.01m`）
