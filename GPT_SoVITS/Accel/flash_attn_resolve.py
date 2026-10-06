"""Resolve a FlashAttention module that exposes flash_attn_with_kvcache.

Order:
1. official / drop-in ``flash_attn`` (Ampere+ or V100 community builds that install as flash_attn)
2. ``flash_attn_v100`` (ai-bond SM70 port), aliased into ``sys.modules['flash_attn']``
"""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
import logging
import sys

import torch


_logger = logging.getLogger(__name__)


def _has_kvcache(module) -> bool:
    return callable(getattr(module, "flash_attn_with_kvcache", None))


def _smoke_kvcache(module, device, dtype) -> bool:
    """Tiny decode-shaped call; catches SM mismatch / broken installs early."""
    try:
        device = torch.device(device)
        bsz, n_head, head_dim, max_len = 1, 2, 64, 16
        q = torch.randn(bsz, 1, n_head, head_dim, device=device, dtype=dtype)
        k_cache = torch.zeros(bsz, max_len, n_head, head_dim, device=device, dtype=dtype)
        v_cache = torch.zeros(bsz, max_len, n_head, head_dim, device=device, dtype=dtype)
        k = torch.randn(bsz, 1, n_head, head_dim, device=device, dtype=dtype)
        v = torch.randn(bsz, 1, n_head, head_dim, device=device, dtype=dtype)
        seqlens = torch.zeros(bsz, device=device, dtype=torch.int32)
        out = module.flash_attn_with_kvcache(
            q, k_cache, v_cache, k, v, cache_seqlens=seqlens
        )
        torch.cuda.synchronize(device)
        return out is not None and out.shape == q.shape
    except Exception as exc:
        _logger.info("flash_attn_with_kvcache smoke test failed: %s", exc)
        return False


@lru_cache(maxsize=1)
def resolve_flash_attn_module():
    """Return a usable flash-attn module, or None."""
    candidates = []
    for name in ("flash_attn", "flash_attn_v100"):
        try:
            mod = import_module(name)
        except (ImportError, OSError, RuntimeError) as exc:
            _logger.info("FlashAttention candidate %s unavailable: %s", name, exc)
            continue
        if _has_kvcache(mod):
            candidates.append((name, mod))

    if not candidates:
        return None

    # Prefer flash_attn if present; otherwise alias flash_attn_v100.
    name, mod = candidates[0]
    if name != "flash_attn":
        sys.modules.setdefault("flash_attn", mod)
        _logger.info("Using %s as flash_attn (SM70 / V100 port)", name)
    return mod


def flash_attn_usable(device, dtype, min_major: int = 7) -> bool:
    """Whether FlashAttention decode backend can run on this device/dtype."""
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    if dtype not in (torch.float16, torch.bfloat16):
        return False
    major, _ = torch.cuda.get_device_capability(device)
    if major < min_major:
        return False

    # Official path: Ampere+.
    # Experimental: SM70 (V100) when a community build is installed.
    if major < 8:
        allow = os_environ_truthy("FLASH_ATTN_V100", default=True)
        if not allow:
            return False

    module = resolve_flash_attn_module()
    if module is None:
        return False

    # Always smoke-test on SM < 8; optional on newer GPUs via env.
    if major < 8 or os_environ_truthy("FLASH_ATTN_SMOKE", default=False):
        return _smoke_kvcache(module, device, dtype)
    return True


def os_environ_truthy(name: str, default: bool = False) -> bool:
    import os

    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "off", "no"}
