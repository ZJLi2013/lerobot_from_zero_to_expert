#!/usr/bin/env bash
# setup_genesis_env.sh
# 在 NV 4090 节点上安装 genesis-world 及依赖
# 用法：bash setup_genesis_env.sh [--check-only]
set -e

CHECK_ONLY=0
if [[ "$1" == "--check-only" ]]; then CHECK_ONLY=1; fi

echo "============================================================"
echo "  Genesis 环境检查 / 安装脚本"
echo "  节点: $(hostname)  |  $(date)"
echo "============================================================"

# ── Python & CUDA 信息 ────────────────────────────────────────────
echo ""
echo "[INFO] Python: $(python --version 2>&1)"
echo "[INFO] pip:    $(pip --version 2>&1 | head -1)"
python -c "import torch; print(f'[INFO] PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "[WARN] torch not importable"

# ── 检查 genesis 是否已安装 ──────────────────────────────────────
echo ""
if python -c "import genesis" 2>/dev/null; then
    GENESIS_VER=$(python -c "import genesis; print(getattr(genesis, '__version__', 'installed'))" 2>/dev/null)
    echo "[OK]  genesis-world 已安装: $GENESIS_VER"
    GENESIS_OK=1
else
    echo "[--]  genesis-world 未安装"
    GENESIS_OK=0
fi

# ── 检查 lerobot ──────────────────────────────────────────────────
if python -c "import lerobot" 2>/dev/null; then
    LEROBOT_VER=$(python -c "import lerobot; print(getattr(lerobot, '__version__', 'installed'))" 2>/dev/null)
    echo "[OK]  lerobot 已安装: $LEROBOT_VER"

    # 定位 SO-101 XML
    XML_PATH=$(python - <<'EOF'
import sys
from pathlib import Path
try:
    import lerobot
    base = Path(lerobot.__file__).parent
    for sub in [
        "common/robot_devices/robots/assets",
        "common/robot_devices/motors/assets",
        "configs/robot",
        ".",
    ]:
        p = base / sub / "so101_new_calib.xml"
        if p.exists():
            print(p)
            sys.exit(0)
    print("NOT_FOUND")
except Exception:
    print("NOT_FOUND")
EOF
    )
    if [[ "$XML_PATH" != "NOT_FOUND" ]]; then
        echo "[OK]  SO-101 XML: $XML_PATH"
    else
        echo "[--]  SO-101 XML 未在 lerobot 包内找到"
    fi
else
    echo "[--]  lerobot 未安装"
fi

if [[ $CHECK_ONLY -eq 1 ]]; then
    echo ""
    echo "[INFO] --check-only 模式，不执行安装"
    exit 0
fi

# ── 检查 Xvfb（headless GPU 服务器必需）────────────────────────────
echo ""
if command -v Xvfb &>/dev/null; then
    echo "[OK]  Xvfb 已安装"
else
    echo "[--]  Xvfb 未安装（headless 服务器需要）"
    if [[ $CHECK_ONLY -eq 0 ]]; then
        echo "[INSTALL] apt-get install xvfb ..."
        apt-get update -qq && apt-get install -y -qq xvfb libgl1 libgles2 libegl1 libglx-mesa0 2>&1 | tail -3
        echo "[OK]  Xvfb + OpenGL libs 安装完成"
    fi
fi

# ── 安装 genesis-world ────────────────────────────────────────────
if [[ $GENESIS_OK -eq 0 ]]; then
    echo ""
    echo "[INSTALL] pip install genesis-world ..."
    pip install genesis-world
    echo "[OK]  genesis-world 安装完成"
else
    echo "[SKIP] genesis-world 已安装，跳过"
fi

echo ""
echo "============================================================"
echo "  设置完成，可以运行 POC："
echo "  python genesis_quick_start.py"
echo ""
echo "  注意：headless 服务器会自动启动 Xvfb，也可手动："
echo "  Xvfb :99 -screen 0 1280x1024x24 -ac &"
echo "  export DISPLAY=:99"
echo "============================================================"
