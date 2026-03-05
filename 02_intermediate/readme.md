# 02 Intermediate Guide

> SO-101 + Genesis 合成数据（SDG）中级实践入口。

---

## 文档分层

- `readme.md`（本文件）：入口与状态总结
- `genesis_sdg.md`：实现流程（环境、脚本、运行步骤）
- `best_practices.md`：调参与验收手册（含最小命令清单）

---

## 本目录关键脚本

- `scripts/sdg_so101_genesis.py`：基础 SO-101 SDG 采集
- `scripts/sdg_so101_improved.py`：改进版（probe + 内置 rrd）
- `scripts/sdg_so101_grasp_experiment.py`：1-episode 调参实验
- `scripts/viz_sdg_rerun.py`：`npy -> .rrd`
- `scripts/npy_to_lerobot.py`：`npy -> LeRobot v3`

---

## 最新验证状态

### NVIDIA 4090（历史）

- 环境：`genesis_poc:latest`
- 状态：`sdg_so101_genesis.py` 可运行并生成 `.npy` / `.rrd`

### AMD MI308X（本轮新增）

- 远端节点：`zhengjli@banff-cyxtera-s71-4.ctr.dcgpu`
- 基础镜像：`rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0`
- 运行模式：Headless（`Xvfb`）
- 结果：`scripts/sdg_so101_genesis.py` **全阶段通过**
  - 1 episode / 180 frames
  - state/action 维度与 `svla_so101_pickplace` 对齐
  - `.rrd` 已回传本地

本地产物：

- `sdg_data/so101_sdg_episode_0_mi308.rrd`

---

## MI308 注意事项（简版）

- 在该基础镜像中，建议安装时固定 `numpy==1.26.4`，避免 Genesis 场景构建的二进制兼容问题
- 脚本默认会自动拉起 `Xvfb`，适配无显示器 headless 远端

---

## 下一步建议

- 若目标是“抓取成功率”，直接转到 `best_practices.md` 做参数迭代
- 若目标是“数据规模化”，按 `genesis_sdg.md` 扩展 episodes 并打包 LeRobot v3

