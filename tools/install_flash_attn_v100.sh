#!/usr/bin/env bash
# Install community FlashAttention for Tesla V100 (SM70).
#
# The upstream ai-bond package pins torch==2.10.0+cu129 (cp312) in pyproject.toml,
# which breaks conda envs like GSV (often Python 3.11 + older CUDA). This script
# strips that pin and builds against the *already installed* torch.
#
# Usage (inside GSV env on AutoDL V100):
#   bash tools/install_flash_attn_v100.sh
#
# Optional:
#   FLASH_ATTN_V100_RELAX_CUDA=1   # skip setup.py CUDA>=12.9 hard check (experimental)
#   FLASH_ATTN_V100=0             # disable FA on SM70 at runtime after install

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${FLASH_ATTN_V100_DIR:-$ROOT/.deps/flash-attention-v100}"
REPO_URL="${FLASH_ATTN_V100_REPO:-https://github.com/ai-bond/flash-attention-v100.git}"

echo "[INFO] Python: $(command -v python)"
python - <<'PY'
import sys
import torch
print(f"[INFO] python={sys.version.split()[0]}")
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
if torch.version.cuda is None:
    raise SystemExit("[ERROR] This torch build has no CUDA")
# Informational only — build may still fail on older toolkits.
ver = tuple(int(x) for x in torch.version.cuda.split(".")[:2])
if ver < (12, 1):
    print(f"[WARN] torch CUDA {torch.version.cuda} is quite old; V100 FA port may fail to compile.")
elif ver < (12, 9):
    print(
        f"[WARN] Upstream setup.py wants CUDA>=12.9 (you have {torch.version.cuda}). "
        "Script will try to relax that check."
    )
PY

if [[ ! -d "$WORKDIR/.git" ]]; then
  echo "[INFO] Cloning $REPO_URL -> $WORKDIR"
  mkdir -p "$(dirname "$WORKDIR")"
  git clone --depth 1 "$REPO_URL" "$WORKDIR"
else
  echo "[INFO] Updating existing clone at $WORKDIR"
  git -C "$WORKDIR" fetch --depth 1 origin || true
  git -C "$WORKDIR" reset --hard origin/main || git -C "$WORKDIR" pull --ff-only || true
fi

cd "$WORKDIR"

echo "[INFO] Patching package metadata to use the current env torch (no cp312 cu129 wheel pin)..."
python - <<'PY'
from pathlib import Path
import re

root = Path(".")
# 1) pyproject.toml: drop hardcoded torch wheel / torch dependency pins
pyproject = root / "pyproject.toml"
if pyproject.exists():
    text = pyproject.read_text(encoding="utf-8")
    # Remove direct torch URL / torch== pins from requires / dependencies lists
    text2 = re.sub(
        r'^\s*["\']torch[^"\']*["\']\s*,?\s*$',
        "",
        text,
        flags=re.M,
    )
    # Also neutralize build-system torch URL requires entries inline
    text2 = re.sub(
        r',\s*["\']torch\s*@[^"\']+["\']',
        "",
        text2,
    )
    text2 = re.sub(
        r'["\']torch\s*@[^"\']+["\']\s*,?',
        "",
        text2,
    )
    if text2 != text:
        pyproject.write_text(text2, encoding="utf-8")
        print("[OK] patched pyproject.toml")
    else:
        print("[INFO] pyproject.toml had no torch URL pin to remove (or already patched)")

# 2) setup.py: relax CUDA>=12.9 guard unless user forbids it
setup = root / "setup.py"
relax = True
import os
if os.environ.get("FLASH_ATTN_V100_RELAX_CUDA", "1").strip() in {"0", "false", "off", "no"}:
    relax = False
if setup.exists() and relax:
    text = setup.read_text(encoding="utf-8")
    text2 = text
    # Common pattern in this repo
    text2 = text2.replace(
        'if parse(torch.version.cuda) < parse("12.9"):',
        'if False and parse(torch.version.cuda) < parse("12.9"):  # relaxed by install_flash_attn_v100.sh',
    )
    text2 = re.sub(
        r'raise RuntimeError\(f?"CUDA version \{torch\.version\.cuda\} < 12\.9 is not supported\."\)',
        'print(f"[WARN] CUDA {torch.version.cuda} < 12.9; continuing with relaxed check")',
        text2,
    )
    if text2 != text:
        setup.write_text(text2, encoding="utf-8")
        print("[OK] relaxed setup.py CUDA>=12.9 check")
    else:
        print("[INFO] setup.py CUDA check not found / already relaxed")
PY

echo "[INFO] Installing build deps (no torch reinstall)..."
python -m pip install -U pip ninja packaging wheel "setuptools>=70"

echo "[INFO] Building flash_attn_v100 against current torch (--no-deps)..."
# Important: --no-deps avoids pulling the cp312 cu129 torch wheel.
# --no-build-isolation uses the env's torch for torch.utils.cpp_extension.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
python -m pip uninstall -y flash-attn flash_attn flash_attn_v100 vllm-flash-attn 2>/dev/null || true
python -m pip install . --no-deps --no-build-isolation -v

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
echo "[DONE] Restart apiV3.py. Expect flash_attn path when smoke test passes."
echo "       Force off:  export FLASH_ATTN_V100=0"
echo "       If build still fails, paste the nvcc/CUDA error block."
