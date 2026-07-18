from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


def load_image_rgb(path: str) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image).astype(np.float32) / 255.0


def load_mask(path: str, threshold: float = 0.5) -> np.ndarray:
    mask = Image.open(path).convert("L")
    array = np.asarray(mask).astype(np.float32) / 255.0
    return (array > threshold).astype(np.uint8)


def resize_to_match(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if source.shape[:2] == target.shape[:2]:
        return source, target
    height, width = target.shape[:2]
    src_uint8 = (np.clip(source, 0.0, 1.0) * 255).astype(np.uint8)
    resized = Image.fromarray(src_uint8).resize((width, height), resample=Image.BILINEAR)
    return np.asarray(resized).astype(np.float32) / 255.0, target


def resize_mask_to(target_hw: Tuple[int, int], mask: np.ndarray) -> np.ndarray:
    height, width = target_hw
    if mask.shape[:2] == (height, width):
        return mask
    resized = Image.fromarray((mask * 255).astype(np.uint8)).resize((width, height), resample=Image.NEAREST)
    return (np.asarray(resized) > 127).astype(np.uint8)


def bbox_from_mask(mask: np.ndarray, pad: int = 0) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(mask.shape[0], y1 + pad)
    x1 = min(mask.shape[1], x1 + pad)
    return int(y0), int(y1), int(x0), int(x1)


def crop(array: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = bbox
    return array[y0:y1, x0:x1]


def apply_mask_rgb(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return image * mask.astype(np.float32)[..., None]


def invert_mask(mask: np.ndarray) -> np.ndarray:
    return (1 - mask).astype(np.uint8)


def union_masks(*masks: np.ndarray | None) -> np.ndarray | None:
    valid = [mask.astype(np.uint8) for mask in masks if mask is not None]
    if not valid:
        return None
    merged = valid[0].copy()
    for mask in valid[1:]:
        merged = np.maximum(merged, mask)
    return (merged > 0).astype(np.uint8)


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path
