#!/bin/bash
# run_poc_docker.sh
#
# 两种用法：
#
# A) 首次（从 pytorch 基础镜像构建 genesis_poc:latest 并运行）：
#   docker run --rm --gpus all \
#     -v ~/github/lerobot_from_zero_to_expert:/workspace/lfzte \
#     pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel \
#     bash /workspace/lfzte/02_intermediate/scripts/run_poc_docker.sh --build
#
# B) 日常 debug（直接用已保存的 genesis_poc:latest）：
#   docker run --rm --gpus all \
#     -v ~/github/lerobot_from_zero_to_expert:/workspace/lfzte \
#     genesis_poc:latest \
#     bash /workspace/lfzte/02_intermediate/scripts/run_poc_docker.sh
#
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_MODE=0
if [ "$1" = "--build" ]; then BUILD_MODE=1; fi

if [ "$BUILD_MODE" = "1" ]; then
    echo "=== [build 1/2] system deps: Xvfb + OpenGL ==="
    apt-get update -qq
    apt-get install -y -qq xvfb libgl1 libgles2 libegl1 libglx-mesa0 2>&1 | tail -3
    echo "deps installed"

    echo "=== [build 2/2] install genesis-world ==="
    pip install genesis-world -q 2>&1 | tail -3
    echo "genesis installed"
fi

echo "=== [run 1/2] start Xvfb :99 (headless display) ==="
Xvfb :99 -screen 0 1280x1024x24 -ac +extension GLX -noreset &
XVFB_PID=$!
export DISPLAY=:99
sleep 2
echo "Xvfb PID=${XVFB_PID}, DISPLAY=${DISPLAY}"

echo "=== [run 2/2] poc_genesis_pipeline.py ==="
cd "${SCRIPT_DIR}"
python poc_genesis_pipeline.py --frames 10 --save /tmp/poc_output

kill "${XVFB_PID}" 2>/dev/null || true
