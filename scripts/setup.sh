#!/usr/bin/env bash
# One-shot environment setup for the CPU benchmark/deploy project.
#   - creates a venv
#   - installs CPU-only torch (no CUDA wheels)
#   - installs strands-decider + serving deps
#   - installs numactl (for core/NUMA pinning) if apt is available
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

PY="${PYTHON:-python3}"
echo ">> creating venv at .venv"
"$PY" -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools -q

echo ">> installing CPU torch (download.pytorch.org/whl/cpu)"
pip install --index-url https://download.pytorch.org/whl/cpu torch -q

echo ">> installing project requirements"
pip install -r requirements.txt -q

echo ">> installing numactl (optional, for pinning)"
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get install -y -q numactl || echo "   (numactl install skipped)"
else
  echo "   apt-get not found; install numactl manually for core pinning"
fi

echo ">> verifying"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "threads", torch.get_num_threads(), "mkldnn", torch.backends.mkldnn.is_available())
import strands_decider
print("strands-decider ok")
PY
echo ">> done. Activate with:  . .venv/bin/activate"
