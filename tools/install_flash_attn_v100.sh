#!/usr/bin/env bash
# Install community FlashAttention for Tesla V100 (SM70).
# Target: Linux + CUDA 12.x + matching PyTorch (see notes below).
#
# Usage (on AutoDL / Linux V100, inside your GSV conda/venv):
#   bash tools/install_flash_attn_v100.sh
#
# After install, restart apiV3.py. Detection uses SM>=7 + smoke test.
# Disable with: export FLASH_ATTN_V100=0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${FLASH_ATTN_V100_DIR:-$ROOT/.deps/flash-attention-v100}"
REPO_URL="${FLASH_ATTN_V100_REPO:-https://github.com/ai-bond/flash-attention-v100.git}"

echo "[INFO] Python: $(command -v python)"
python - <<'PY'
import torch
print(f"[INFO] torch={torch.__version__} cuda={torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("[ERROR] CUDA not available in this Python")
for i in range(torch.cuda.device_count()):
    major, minor = torch.cuda.get_device_capability(i)
    name = torch.cuda.get_device_name(i)
    print(f"[INFO] GPU{i}: {name} SM {major}.{minor}")
    if major < 7:
        raise SystemExit("[ERROR] GPU older than Volta; FlashAttention V100 port not applicable")
    if major >= 8:
        print("[WARN] This GPU is Ampere+. Prefer official flash-attn instead of the V100 port.")
PY

if [[ ! -d "$WORKDIR/.git" ]]; then
  echo "[INFO] Cloning $REPO_URL -> $WORKDIR"
  mkdir -p "$(dirname "$WORKDIR")"
  git clone --depth 1 "$REPO_URL" "$WORKDIR"
else
  echo "[INFO] Updating existing clone at $WORKDIR"
  git -C "$WORKDIR" pull --ff-only || true
fi

cd "$WORKDIR"

echo "[INFO] Installing build deps (ninja/packaging)..."
python -m pip install -U pip ninja packaging wheel setuptools

echo "[INFO] Building flash_attn_v100 (this can take a long time)..."
# Prefer project installer if present
if [[ -x ./run.sh ]]; then
  ./run.sh
else
  python -m pip install . --no-build-isolation -v
fi

echo "[INFO] Verifying imports..."
python - <<'PY'
import torch
from importlib import import_module

mod = None
for name in ("flash_attn", "flash_attn_v100"):
    try:
        mod = import_module(name)
        print(f"[OK] import {name}: {getattr(mod, '__doc__', '')!r[:80]}")
        break
    except Exception as e:
        print(f"[MISS] {name}: {e}")
if mod is None or not callable(getattr(mod, "flash_attn_with_kvcache", None)):
    raise SystemExit("[ERROR] flash_attn_with_kvcache missing after install")

device = torch.device("cuda")
dtype = torch.float16
bsz, n_head, head_dim, max_len = 1, 2, 64, 16
q = torch.randn(bsz, 1, n_head, head_dim, device=device, dtype=dtype)
k_cache = torch.zeros(bsz, max_len, n_head, head_dim, device=device, dtype=dtype)
v_cache = torch.zeros(bsz, max_len, n_head, head_dim, device=device, dtype=dtype)
k = torch.randn(bsz, 1, n_head, head_dim, device=device, dtype=dtype)
v = torch.randn(bsz, 1, n_head, head_dim, device=device, dtype=dtype)
seqlens = torch.zeros(bsz, device=device, dtype=torch.int32)
out = mod.flash_attn_with_kvcache(q, k_cache, v_cache, k, v, cache_seqlens=seqlens)
torch.cuda.synchronize()
assert out.shape == q.shape
print("[SUCCESS] flash_attn_with_kvcache smoke test passed on", torch.cuda.get_device_name(0))
PY

echo
echo "[DONE] Restart apiV3.py. Expect logs to enable flash_attn when supported."
echo "       Force off:  export FLASH_ATTN_V100=0"
echo "       Force smoke: export FLASH_ATTN_SMOKE=1"
