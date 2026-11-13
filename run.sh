#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-python3}"
if [[ "${1:-}" == "setup" ]]; then
  "$PYTHON" -c 'import sys; assert sys.version_info >= (3,11), "Python >= 3.11 required"; import torch; assert torch.cuda.is_available(), "Install a CUDA-enabled PyTorch build first: https://pytorch.org/get-started/locally/"'
  "$PYTHON" -m pip install -r requirements.txt
  exit 0
fi
mkdir -p logs
# Preserve errors across tee; use distinct logs on every invocation.
"$PYTHON" -u launch.py "$@" 2>&1 | tee "logs/launch_$(date +%Y%m%d_%H%M%S)_$$.log"
