from __future__ import annotations

import os


def compute_fid(real_dir: str, generated_dir: str, device: str = "cuda", batch_size: int = 64, mode: str = "clean") -> float:
    if not os.path.isdir(real_dir) or not os.path.isdir(generated_dir):
        raise FileNotFoundError(f"FID directories not found: real={real_dir}, generated={generated_dir}")
    try:
        from cleanfid import fid as cleanfid_fid  # type: ignore
    except Exception as exc:
        raise ImportError("clean-fid is required for FID. Install it with `pip install clean-fid`.") from exc
    return float(cleanfid_fid.compute_fid(real_dir, generated_dir, device=device, batch_size=batch_size, mode=mode))
