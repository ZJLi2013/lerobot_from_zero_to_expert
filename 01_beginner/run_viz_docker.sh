#!/bin/bash
# LeRobot 数据集可视化 Docker 启动脚本
# 用法:
#   bash 01_beginner/run_viz_docker.sh [mode] [episode]
#
# 示例:
#   bash 01_beginner/run_viz_docker.sh stats 0        # 纯统计模式（默认）
#   bash 01_beginner/run_viz_docker.sh web   0        # Web Viewer（浏览器访问）
#   bash 01_beginner/run_viz_docker.sh save  0        # 保存 .rrd 文件
#
# 在远端节点运行（需同时 clone lerobot 和 lerobot_from_zero_to_expert）
# Web 模式运行后，本地浏览器访问: http://<4090_HOST>:9090
# 或 SSH 转发后访问:              http://localhost:9090

MODE=${1:-stats}
EPISODE=${2:-0}

echo "=== LeRobot 数据集可视化 ==="
echo "模式: $MODE  |  Episode: $EPISODE"
echo ""

if [ "$MODE" = "web" ]; then
    # Web 模式：需要暴露端口
    docker run --rm --gpus all \
      -v ~/github/lerobot:/workspace/lerobot \
      -v ~/github/lerobot_from_zero_to_expert:/workspace/tutorial \
      -v ~/hf_cache:/root/.cache/huggingface \
      -v ~/lerobot_viz_output:/workspace/lerobot/outputs/viz \
      -w /workspace/lerobot \
      -p 9090:9090 \
      -p 9876:9876 \
      pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel \
      bash -c "pip install -e '.' -q \
               && pip uninstall torchcodec -y -q \
               && pip install rerun-sdk -q \
               && python /workspace/tutorial/01_beginner/viz_dataset_so101.py \
                    --mode web \
                    --episode-index ${EPISODE} \
                    --web-port 9090 \
                    --grpc-port 9876"
elif [ "$MODE" = "save" ]; then
    docker run --rm --gpus all \
      -v ~/github/lerobot:/workspace/lerobot \
      -v ~/github/lerobot_from_zero_to_expert:/workspace/tutorial \
      -v ~/hf_cache:/root/.cache/huggingface \
      -v ~/lerobot_viz_output:/workspace/lerobot/outputs/viz \
      -w /workspace/lerobot \
      pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel \
      bash -c "pip install -e '.' -q \
               && pip uninstall torchcodec -y -q \
               && pip install rerun-sdk -q \
               && python /workspace/tutorial/01_beginner/viz_dataset_so101.py \
                    --mode save \
                    --episode-index ${EPISODE} \
                    --output-dir outputs/viz"
else
    # stats 模式（默认）
    docker run --rm --gpus all \
      -v ~/github/lerobot:/workspace/lerobot \
      -v ~/github/lerobot_from_zero_to_expert:/workspace/tutorial \
      -v ~/hf_cache:/root/.cache/huggingface \
      -v ~/lerobot_viz_output:/workspace/lerobot/outputs/viz \
      -w /workspace/lerobot \
      pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel \
      bash -c "pip install -e '.' -q \
               && pip uninstall torchcodec -y -q \
               && pip install matplotlib -q \
               && python /workspace/tutorial/01_beginner/viz_dataset_so101.py \
                    --mode stats \
                    --episode-index ${EPISODE} \
                    --output-dir outputs/viz"
fi
