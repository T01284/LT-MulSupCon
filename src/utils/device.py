"""Device selection with CUDA compatibility probe."""

from __future__ import annotations

import torch


def probe_cuda() -> bool:
    """Return True only if CUDA kernels can actually run on this GPU."""
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.randn(8, 8, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        _ = float(y.sum())
        return True
    except RuntimeError:
        return False


def resolve_device(preference: str = "auto") -> torch.device:
    pref = (preference or "auto").lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if probe_cuda():
            return torch.device("cuda")
        raise RuntimeError(
            "CUDA was requested but is unavailable or incompatible with this PyTorch build."
        )
    if probe_cuda():
        return torch.device("cuda")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(
            f"[device] GPU detected ({name}, sm_{cap[0]}{cap[1]}) "
            f"but PyTorch {torch.__version__} cannot run CUDA kernels; using CPU."
        )
    else:
        print("[device] CUDA not available; using CPU.")
    return torch.device("cpu")
