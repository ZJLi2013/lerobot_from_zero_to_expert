# LeRobot Quick-Start 测试报告

**测试日期**: 2026-03-04 / 2026-03-05  
**代码仓库**: `https://github.com/ZJLi2013/lerobot_from_zero_to_expert` (branch: `main`)

> 已在两种硬件平台验证通过 —— NVIDIA RTX 4090 和 AMD Instinct MI308X

---

## 1. 环境信息

### 1.1 NVIDIA 节点

| 项目 | 值 |
|------|-----|
| Docker 镜像 | `pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel` |
| Python | 3.11.13 (conda-forge) |
| PyTorch | 2.8.0+cu126 |
| CUDA | 12.6.3 |
| GPU | 2 × NVIDIA GeForce RTX 4090 |
| 单卡显存 | 24 GB |
| 视频解码后端 | pyav（torchcodec 与 PT 2.8.0 ABI 不兼容，需卸载）|

### 1.2 AMD 节点

| 项目 | 值 |
|------|-----|
| Docker 镜像 | `rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0` |
| Python | 3.12 |
| PyTorch | 2.6.0+rocm6.4.3 |
| ROCm | 6.4.3 |
| GPU | 8 × AMD Instinct MI308X |
| 单卡显存 | 192 GB HBM3 |
| 视频解码后端 | pyav（同样需要 `pip uninstall torchcodec`）|

---

## 2. 测试结果汇总

| 测试脚本 | NVIDIA RTX 4090 | AMD MI308X | 备注 |
|----------|----------------|-----------|------|
| `test_diffusion_pusht.py` | **全部通过 ✓** | **全部通过 ✓** | 6/6 步 PASS |
| `test_act_so101.py` | **全部通过 ✓** | **全部通过 ✓** | 6/6 步 PASS |
| `test_smolvla_so101.py` | **全部通过 ✓** | **全部通过 ✓** | 6/6 步 PASS |
| `viz_dataset_so101.py` | **全部通过 ✓** | 未测试 | stats/save 两种模式均正常 |

---

## 3. 详细测试结果

### 3.1 test_diffusion_pusht.py — Diffusion Policy + PushT

**数据集**: `lerobot/pusht`（HuggingFace，25,650 帧，10 FPS）  
**策略**: Diffusion Policy（UNet 骨干）

| 步骤 | 内容 | 结果 |
|------|------|------|
| [1/5] 模块导入 | lerobot, diffusion, datasets | PASS ✓ |
| [2/5] 数据集元数据 | pusht: 25650帧, FPS=10, 输入=[image, state], 输出=[action] | PASS ✓ |
| [3/5] 策略创建 | 参数量: 262,709,026, device=cuda | PASS ✓ |
| [4/5] DataLoader | 25650 样本, batch_size=8 | PASS ✓ |
| [5/5] 训练循环 (5步) | Loss 趋势见下表 | PASS ✓ |
| [Extra] 模型保存 | `outputs/test/pusht_diffusion_test/` | PASS ✓ |

**训练 Loss（5步）**:

| Step | Loss |
|------|------|
| 1 | 0.9578 |
| 2 | 4.4149 |
| 3 | 1.6086 |
| 4 | 1.2542 |
| 5 | 1.0536 |

**峰值显存**: 5058 MB（**4.94 GB**）

> Diffusion Policy 的 Loss 在早期可能有波动（step 2 从 0.96 跳到 4.41），这是 diffusion 模型训练初期的正常现象——噪声预测随机性导致单步 loss 不单调，不代表训练异常。

---

### 3.2 test_act_so101.py — ACT + SO-101 真实机器人数据

**数据集**: `lerobot/svla_so101_pickplace`（HuggingFace，11,939 帧，30 FPS）  
**策略**: ACT（Action Chunking Transformer）  
**backbone**: ResNet-18（自动下载预训练权重 44.7 MB）

| 步骤 | 内容 | 结果 |
|------|------|------|
| [1/5] 模块导入 | lerobot, act, datasets | PASS ✓ |
| [2/5] 数据集元数据 | svla_so101_pickplace: 11939帧, FPS=30, 输入=[state, up_cam, side_cam], 输出=[action] | PASS ✓ |
| [3/5] 策略创建 | 参数量: 28,775,046, chunk_size=100, dim_model=256, device=cuda | PASS ✓ |
| [4/5] DataLoader | 11939 样本, action δt=100 步, batch_size=4 | PASS ✓ |
| [5/5] 训练循环 (5步) | Loss 趋势见下表 | PASS ✓ |
| [Extra] 模型保存 | `outputs/test/act_so101_test/` | PASS ✓ |

**训练 Loss（5步）**:

| Step | Total Loss | kl loss | l1 loss |
|------|-----------|---------|---------|
| 1 | 78.7325 | — | — |
| 2 | 76.0214 | — | — |
| 3 | 67.2848 | — | — |
| 4 | 53.2243 | — | — |
| 5 | 47.3565 | — | — |

**峰值显存**: 1810 MB（**1.77 GB**）

> `kl=nan, l1=nan`：`info` 字典中未单独记录 kl/l1 分量（测试脚本用 `info.get(..., nan)` 兜底），但 total loss 正常且持续下降（78.7 → 47.4），训练 pipeline 正常。ACT 默认配置将 kl loss 归入 total loss，分量追踪需显式开启。

### 3.3 test_smolvla_so101.py — SmolVLA + SO-101 真实机器人数据

**数据集**: `lerobot/svla_so101_pickplace`（HuggingFace，11,939 帧，30 FPS，含语言任务描述）  
**策略**: SmolVLA（Vision-Language-Action，450M 参数）  
**backbone**: SmolVLM2-500M-Video-Instruct（视觉语言模型）+ Flow Matching 动作专家  
**预训练权重**: `lerobot/smolvla_base`（从 HuggingFace Hub 加载，耗时 ~44.6s）  
**微调策略**: `freeze_vision_encoder=True`, `train_expert_only=True`（仅训练动作专家 + 状态投影层）

| 步骤 | 内容 | 结果 |
|------|------|------|
| [1/6] 模块导入 | lerobot, smolvla, datasets | PASS ✓ |
| [2/6] 数据集元数据 | svla_so101_pickplace: 11939帧, FPS=30, 50 episodes, 含语言描述 | PASS ✓ |
| [3/6] 策略创建 | 总参数 ~450M, 可训练 ~99.9M, 冻结 ~350M, device=cuda | PASS ✓ |
| [4/6] DataLoader | 11939 样本, chunk_size=50, batch_size=2 | PASS ✓ |
| [5/6] 训练循环 (5步) | Loss 趋势见下表 | PASS ✓ |
| [6/6] 模型保存 | `outputs/test/smolvla_so101_test/` | PASS ✓ |

**训练 Loss（5步）**:

| Step | Loss | 耗时 |
|------|------|------|
| 1 | 703.9158 | 0.75s |
| 2 | 508.2873 | 0.09s |
| 3 | 727.9923 | 0.11s |
| 4 | 743.7834 | 0.11s |
| 5 | 427.7461 | 0.11s |

**峰值显存**: 2017 MB（**1.97 GB**）

> SmolVLA 的 Flow Matching Loss 初期波动属正常现象（随机噪声时间步 t 导致 loss 不单调）。训练 5 步的目的仅为验证 pipeline 通路，非验证收敛性。预训练权重加载时 `strict=False`，状态/动作投影层因数据集不同而重新初始化，VLM 骨干权重完整复用。

---

### 3.4 viz_dataset_so101.py — 数据集可视化（Rerun）

**数据集**: `lerobot/svla_so101_pickplace`（11,939 帧，50 episodes，双摄像头）  
**工具**: Rerun SDK 0.26.2 + Matplotlib  
**测试模式**: `stats`（纯统计，headless）+ `save`（生成 .rrd 文件）

| 步骤 | 内容 | 结果 |
|------|------|------|
| stats 模式 | 生成 action/state 时序曲线图（PNG）+ 相机截帧图 | PASS ✓ |
| save 模式 | 生成 `lerobot_svla_so101_pickplace_episode_0.rrd`（117 MB）| PASS ✓ |

**Stats 模式输出文件**（保存至 `~/lerobot_viz_output/`）：

| 文件 | 大小 | 内容 |
|------|------|------|
| `action_episode_0.png` | 122 KB | 6维动作序列时序曲线（Episode 0，303帧，10.07s）|
| `state_episode_0.png` | 112 KB | 6维关节状态时序曲线 |
| `frames_episode_0.png` | 224 KB | 双摄像头截帧（每5秒一帧，up + side）|
| `lerobot_svla_so101_pickplace_episode_0.rrd` | **117 MB** | Rerun 归档文件，本地 `rerun *.rrd` 打开 |

**Episode 0 数据统计**：
- 帧数：303 帧 / 时长：10.07 秒 / FPS：30
- Action 均值 (6D): `[18.76, -58.75, 66.51, 69.60, -48.33, 5.95]`
- State 均值 (6D):  `[18.74, -57.78, 67.74, 69.70, -48.27, 6.89]`

**Headless 可视化三种方案**（详见 `data_visualize.md`）：

| 方案 | 命令 | 访问方式 |
|------|------|---------|
| **Web 浏览器** | `run_viz_docker.sh web 0` | `http://<4090_HOST>:9090` 或 SSH 转发后 `localhost:9090` |
| **保存 .rrd** | `run_viz_docker.sh save 0` | `scp *.rrd` 到本地，`rerun *.rrd` 打开 |
| **纯统计** | `run_viz_docker.sh stats 0` | 生成 PNG，`scp` 到本地查看 |

> Web 模式需要在 `docker run` 时额外指定 `-p 9090:9090 -p 9876:9876` 暴露端口，已封装在 `run_viz_docker.sh` 中。

---

## 4. 兼容性问题与修复

### 问题：torchcodec 0.10.0 与 PyTorch 2.8.0 ABI 不兼容

**现象**：lerobot 0.4.5 默认安装 `torchcodec 0.10.0`，DataLoader 在 worker 进程加载视频帧时抛出：

```
RuntimeError: Could not load libtorchcodec.
OSError: /opt/conda/.../libtorchcodec_core4.so: undefined symbol: _ZN3c1013MessageLogger6streamB5cxx11Ev
```

**根因**：torchcodec 0.10.0 使用的 libtorch ABI 符号（`B5cxx11` 表示 C++11 dual ABI）与 PyTorch 2.8.0 编译版本不匹配。

**修复方案**：在容器内安装完 lerobot 后，卸载 torchcodec，让 lerobot 自动降级到内置的 `pyav` 后端（PyAV 自带 FFmpeg，无需系统级安装）：

```bash
pip install -e ".[pusht]"
pip uninstall torchcodec -y
```

lerobot 会输出：
```
WARNING: 'torchcodec' is not available, falling back to 'pyav' as a default decoder
```

之后 DataLoader 正常运行。

**长期修复建议**：
- 等待 torchcodec 发布支持 PyTorch 2.8.0 的新版本
- 或在 lerobot 安装时锁定 `torchcodec` 版本范围
- 或显式传入 `video_backend="pyav"` 参数给 `LeRobotDataset`

### 其他 Warning（不影响结果）

```
UserWarning: The video decoding capabilities of torchvision are deprecated from 0.22
and will be removed in 0.24. Migrate to TorchCodec.
```
这是 torchvision pyav 后端的废弃提示，不影响本次测试。

---

## 5. 完整操作流程（Step by Step）

> 本流程记录了从本地开发到远端 Docker 测试的完整闭环，可复用于其他节点。

### Step 1：配置 SSH 密钥登录（一次性）

Windows 上 `ssh` 不支持命令行直传密码，需先通过 paramiko 把本地公钥写入远端：

```python
# 使用 paramiko 密码登录，拷贝 ~/.ssh/id_ed25519.pub → 远端 authorized_keys
import paramiko, os
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('<4090_HOST>', username='david', password='<password>')
pub_key = open(os.path.expanduser('~/.ssh/id_ed25519.pub')).read().strip()
client.exec_command(f'mkdir -p ~/.ssh && echo "{pub_key}" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys')
client.close()
```

完成后即可免密 `ssh david@<4090_HOST>`。

### Step 2：本地修改代码并推送到个人 fork

```powershell
cd C:\Users\zhengjli\Documents\github\lerobot_from_zero_to_expert
git add 01_beginner/
git commit -m "Add beginner tutorial scripts"
git pull --rebase origin main
git push origin main
```

### Step 3：远端克隆仓库（首次）

```bash
ssh david@<4090_HOST>
cd ~/github
# lerobot 主库（用于 pip install）
git clone https://github.com/huggingface/lerobot.git
# 教程脚本库
git clone https://github.com/<your>/lerobot_from_zero_to_expert.git
```

后续更新只需：
```bash
ssh david@<4090_HOST> "cd ~/github/lerobot_from_zero_to_expert && git pull --ff-only"
```

### Step 4：检查 Docker 镜像

```bash
docker images pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel
# 输出：pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel   14GB
```

若未拉取：`docker pull pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel`

### Step 5：在 Docker 容器内安装 lerobot + 运行测试

每次测试在**全新容器**（`--rm`）中运行，保证环境干净。关键点：
- 挂载 `~/github/lerobot` 到 `/workspace/lerobot`（代码）
- 挂载 `~/hf_cache` 到 `/root/.cache/huggingface`（数据集缓存，避免重复下载）
- **安装后卸载 torchcodec**（修复 PyTorch 2.8.0 ABI 兼容问题）

**Diffusion Policy + PushT**：
```bash
docker run --rm --gpus all \
  -v ~/github/lerobot:/workspace/lerobot \
  -v ~/github/lerobot_from_zero_to_expert:/workspace/tutorial \
  -v ~/hf_cache:/root/.cache/huggingface \
  -w /workspace/lerobot \
  pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel \
  bash -c 'pip install -e ".[pusht]" -q \
           && pip uninstall torchcodec -y -q \
           && python /workspace/tutorial/01_beginner/test_diffusion_pusht.py'
```

**ACT + SO-101**（lerobot 无单独 `[act]` extra，直接装基础包）：
```bash
docker run --rm --gpus all \
  -v ~/github/lerobot:/workspace/lerobot \
  -v ~/github/lerobot_from_zero_to_expert:/workspace/tutorial \
  -v ~/hf_cache:/root/.cache/huggingface \
  -w /workspace/lerobot \
  pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel \
  bash -c 'pip install -e . -q \
           && pip uninstall torchcodec -y -q \
           && python /workspace/tutorial/01_beginner/test_act_so101.py'
```

**SmolVLA + SO-101**（需 `[smolvla]` extra，含 transformers/accelerate 依赖）：
```bash
docker run --rm --gpus all \
  -v ~/github/lerobot:/workspace/lerobot \
  -v ~/github/lerobot_from_zero_to_expert:/workspace/tutorial \
  -v ~/hf_cache:/root/.cache/huggingface \
  -w /workspace/lerobot \
  pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel \
  bash -c 'pip install -e ".[smolvla]" -q \
           && pip uninstall torchcodec -y -q \
           && python /workspace/tutorial/01_beginner/test_smolvla_so101.py'
```

> 或直接运行仓库内的封装脚本：`bash 01_beginner/run_smolvla_docker.sh`

### Step 6：数据集下载说明

测试首次运行会自动从 HuggingFace 下载数据集：

| 数据集 | 大小（元数据） | 首次下载 |
|--------|--------------|---------|
| `lerobot/pusht` | 25,650 帧, ~几百 MB | 自动，约 5-10 秒（有缓存后秒开） |
| `lerobot/svla_so101_pickplace` | 11,939 帧, ~几百 MB | 自动，约 5-10 秒（有缓存后秒开） |

数据缓存在 `~/hf_cache`（通过 volume 挂载持久化），重复运行不再下载。

ACT 策略还会自动下载 ResNet-18 预训练权重（44.7 MB）：
```
Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth"
```

SmolVLA 策略从 `lerobot/smolvla_base` 下载完整预训练权重（450M 模型，约数 GB）：
首次下载较慢，建议挂载 `~/hf_cache` 持久化缓存。加载时间约 44.6s（含模型初始化）。

### Step 7：验收标准

每个测试脚本退出码为 0，且末尾打印：
```
✓ 所有测试通过！
```

---

## 6. 性能对比

### 6.1 NVIDIA RTX 4090（PyTorch 2.8.0 + CUDA 12.6）

| 指标 | Diffusion Policy | ACT | SmolVLA |
|------|-----------------|-----|---------|
| 模型参数量 | ~263M | ~29M | ~450M（可训练 ~100M） |
| 峰值显存 | 4.94 GB | 1.77 GB | **1.97 GB** |
| batch_size | 8 | 4 | 2 |
| 单步耗时 | ~12s/步 | ~6s/步 | 0.75s（step1）/ ~0.1s（后续） |
| Loss 趋势 | 0.96→4.41→1.61→1.25→1.05 | 78.7→47.4（单调下降） | 703→508→728→744→428 |

### 6.2 AMD MI308X（PyTorch 2.6.0 + ROCm 6.4.3）

| 指标 | Diffusion Policy | ACT | SmolVLA |
|------|-----------------|-----|---------|
| 峰值显存 | **4.92 GB** | **1.83 GB** | **2.02 GB** |
| batch_size | 8 | 4 | 2 |
| 单步耗时 | — | — | 25.1s（step1）/ ~0.3s（后续） |
| Loss 趋势 | 1.17→5.49→1.17→1.48→1.24 | 73.5→76.4→63.0→58.1→42.5（下降）| 437→796→845→546→486 |
| Docker GPU flags | `--gpus all` | `--gpus all` | `--gpus all` |

> **AMD ROCm vs NVIDIA CUDA 说明**：PyTorch ROCm 版本将 `torch.cuda.*` API 全部映射到 HIP/ROCm，Python 代码无需修改。  
> Docker 启动方式不同：CUDA 用 `--gpus all`，ROCm 用 `--device /dev/kfd --device /dev/dri --group-add video`。

### 6.3 策略架构说明

| 项目 | Diffusion Policy | ACT | SmolVLA |
|------|-----------------|-----|---------|
| backbone | UNet（from scratch） | ResNet-18（pretrained）+ Transformer | SmolVLM2-500M（冻结）+ Flow Matching Expert |
| pip extra | `.[pusht]` | `.` | `.[smolvla]` |
| 预训练来源 | 无 | ResNet-18 (44.7 MB) | lerobot/smolvla_base (~450M，加载 ~44.6s) |

> SmolVLA 显存极低（2 GB 以内）：`freeze_vision_encoder=True` + `train_expert_only=True`，骨干 VLM（~350M 参数）冻结无梯度。

---

## 7. 结论

- **全平台验证通过**：三个训练 demo 脚本在 NVIDIA RTX 4090（CUDA）和 AMD Instinct MI308X（ROCm）两种平台均 **18/18 步 PASS**
- **跨平台兼容性**：PyTorch ROCm 版本保持 `torch.cuda.*` API 接口，Python 脚本无需任何修改即可在 AMD GPU 上运行
- **关键兼容性修复**：两个平台均需卸载 `torchcodec`（libavdevice.so 缺失），改用 `pyav` 后端
- **SmolVLA 亮点**：450M VLA 模型微调仅需 **~2 GB 显存**（冻结 VLM 骨干），消费级 GPU 与数据中心 GPU 均可运行
- **显存对比**：两平台显存占用基本一致（Diffusion ~5 GB、ACT ~1.8 GB、SmolVLA ~2 GB）
- **Docker 差异**：NVIDIA 用 `--gpus all`，AMD ROCm 用 `--device /dev/kfd --device /dev/dri --group-add video`
