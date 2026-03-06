# 02 Intermediate Guide

> SO-101 + Genesis 合成数据（SDG）中级实践入口。

---

## 文档分层

- `readme.md`（本文件）：入口与状态总结
- `doc/implement_guide.md`：实现流程（环境、脚本、运行步骤）
- `doc/best_practices.md`：调参与验收手册（含最小命令清单）

---

## 本目录关键脚本

- `scripts/1_poc_pipeline.py`：Genesis POC 验证管线
- `scripts/2_collect.py`：SO-101 采集（probe + 朝向修正 + .rrd 输出）
- `scripts/3_grasp_experiment.py`：1-episode 调参实验
- `scripts/4_parallel_lerobot.py`：并行批量采集 + 直接写 LeRobot 格式
- `scripts/viz_sdg_rerun.py`：`npy -> .rrd`
- `scripts/npy_to_lerobot.py`：`npy -> LeRobot v3`

---

## 最新验证状态

### NVIDIA 4090

- 基础镜像：`pytorch/pytorch:2.9.1-cuda12.6-cudnn9-devel`
- 运行模式：Headless（`Xvfb`）
- 结果: **全阶段通过**

### AMD MI308X

- 基础镜像：`rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0`
- 运行模式：Headless（`Xvfb`）
- 结果： **全阶段通过**

**MI308 注意事项**
- 在该基础镜像中，建议安装时固定 `numpy==1.26.4`，避免 Genesis 场景构建的二进制兼容问题
- 脚本默认会自动拉起 `Xvfb`，适配无显示器 headless 远端

---


## 下一步建议

- 若目标是“抓取成功率”，直接转到 `best_practices.md` 做参数迭代
- 若目标是“数据规模化”，按 `implement_guide.md` 扩展 episodes 并打包 LeRobot v3

* [add domain randomize](https://genesis-world.readthedocs.io/zh-cn/latest/user_guide/getting_started/domain_randomization.html)