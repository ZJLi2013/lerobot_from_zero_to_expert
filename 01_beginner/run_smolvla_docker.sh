#!/bin/bash
# SmolVLA 训练测试 Docker 启动脚本
# 用法: bash 01_beginner/run_smolvla_docker.sh
# 在远端节点运行（需同时 clone lerobot 和 lerobot_from_zero_to_expert）
#   远端准备:
#     cd ~/github && git clone https://github.com/huggingface/lerobot
#     cd ~/github && git clone https://github.com/<your>/lerobot_from_zero_to_expert

docker run --rm --gpus all \
  -v ~/github/lerobot:/workspace/lerobot \
  -v ~/github/lerobot_from_zero_to_expert:/workspace/tutorial \
  -v ~/hf_cache:/root/.cache/huggingface \
  -w /workspace/lerobot \
  pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel \
  bash -c 'pip install -e ".[smolvla]" -q && pip uninstall torchcodec -y -q && python /workspace/tutorial/01_beginner/test_smolvla_so101.py'
