from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_fn

try:
    import lpips  # type: ignore
except Exception:
    lpips = None


def masked_mse(x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> float:
    squared_error = (x - y) ** 2
    if mask is None:
        return float(squared_error.mean())
    valid = mask.astype(bool)
    if valid.sum() == 0:
        return float("nan")
    return float(squared_error[valid].mean())


def psnr_from_mse(mse: float, max_val: float = 1.0) -> float:
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10((max_val ** 2) / mse))


def psnr(x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> float:
    return psnr_from_mse(masked_mse(x, y, mask))


def ssim_rgb(x: np.ndarray, y: np.ndarray) -> float:
    return float(ssim_fn(x, y, channel_axis=-1, data_range=1.0))


class LPIPSEngine:
    def __init__(self, device: str = "cuda"):
        if lpips is None:
            raise RuntimeError("lpips is not installed. Install it with `pip install lpips`.")
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.model = lpips.LPIPS(net="alex").to(self.device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        tx = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)
        ty = torch.from_numpy(y).permute(2, 0, 1).unsqueeze(0).to(self.device)
        tx = tx * 2 - 1
        ty = ty * 2 - 1
        return float(self.model(tx, ty).item())
