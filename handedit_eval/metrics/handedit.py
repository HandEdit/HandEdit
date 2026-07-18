from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from skimage.color import rgb2hsv, rgb2lab

from .generic import LPIPSEngine


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float32)
    norm = float(np.linalg.norm(vector)) + 1e-8
    return vector / norm


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(_l2_normalize(a), _l2_normalize(b)), -1.0, 1.0))


def _cosine01(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * (1.0 + _cosine(a, b))


def _harmonic_mean(a: float, b: float) -> float:
    a = float(np.clip(a, 0.0, 1.0))
    b = float(np.clip(b, 0.0, 1.0))
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return 2.0 * a * b / (a + b + 1e-8)


def _skin_mask(image: np.ndarray) -> np.ndarray:
    hsv = rgb2hsv(np.clip(image, 0.0, 1.0))
    hue = hsv[..., 0]
    sat = hsv[..., 1]
    val = hsv[..., 2]
    mask = ((hue <= 0.12) | (hue >= 0.90)) & (sat >= 0.20) & (sat <= 0.75) & (val >= 0.20) & (val <= 0.98)
    return mask.astype(np.uint8)


def _bbox_from_mask(mask: np.ndarray, pad: int = 8) -> Optional[tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(mask.shape[0], int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(mask.shape[1], int(xs.max()) + 1 + pad)
    return y0, y1, x0, x1


def _crop_to_bbox(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = bbox
    return image[y0:y1, x0:x1]


def _crop_to_mask(image: np.ndarray, mask: np.ndarray, pad: int = 8) -> np.ndarray:
    bbox = _bbox_from_mask(mask, pad=pad)
    return image if bbox is None else _crop_to_bbox(image, bbox)


def _masked_crop_to_mask(image: np.ndarray, mask: np.ndarray, pad: int = 8, fill_value: float = 0.5) -> np.ndarray:
    """Crop the foreground bbox and paint the outside with a neutral color."""
    bbox = _bbox_from_mask(mask, pad=pad)
    if bbox is None:
        return image
    crop_image = _crop_to_bbox(image, bbox)
    crop_mask = _crop_to_bbox(mask.astype(np.uint8), bbox).astype(bool)
    fill = np.full_like(crop_image, float(fill_value), dtype=np.float32)
    return np.where(crop_mask[..., None], crop_image, fill)


def _dilate_mask(mask: np.ndarray, width: int) -> np.ndarray:
    mask_bool = mask.astype(bool)
    if width <= 0:
        return mask_bool.astype(np.uint8)
    return binary_dilation(mask_bool, iterations=max(1, int(width))).astype(np.uint8)


def _ensure_min_size(image: np.ndarray, min_size: int = 64) -> np.ndarray:
    height, width = image.shape[:2]
    if height >= min_size and width >= min_size:
        return image
    scale = float(min_size) / float(max(1, min(height, width)))
    new_width = max(min_size, int(round(width * scale)))
    new_height = max(min_size, int(round(height * scale)))
    pil_image = Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8))
    pil_image = pil_image.resize((new_width, new_height), Image.BICUBIC)
    return np.asarray(pil_image).astype(np.float32) / 255.0


def _resize_mask(mask: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    height, width = hw
    if mask.shape[:2] == (height, width):
        return (mask > 0).astype(np.uint8)
    pil = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    pil = pil.resize((width, height), Image.NEAREST)
    return (np.asarray(pil) > 127).astype(np.uint8)


def _resize_image_and_mask(image: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    image_u8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    img = Image.fromarray(image_u8).resize((size, size), Image.BILINEAR)
    msk = Image.fromarray((mask > 0).astype(np.uint8) * 255).resize((size, size), Image.NEAREST)
    return np.asarray(img).astype(np.float32) / 255.0, (np.asarray(msk) > 127).astype(np.uint8)


def _full_mask(hw: tuple[int, int]) -> np.ndarray:
    return np.ones(hw, dtype=np.uint8)


def _contact_band(roi_mask: np.ndarray, width: int) -> np.ndarray:
    dilated = binary_dilation(roi_mask.astype(bool), iterations=max(1, int(width)))
    band = np.logical_and(dilated, ~roi_mask.astype(bool))
    return band.astype(np.uint8)


def _masked_lab_similarity(
    pred_image: np.ndarray,
    pred_mask: np.ndarray,
    gt_image: np.ndarray,
    gt_mask: np.ndarray,
    pad: int,
    size: int,
    tau: float,
) -> tuple[float, float]:
    """Pixel-wise CIE Lab distance after foreground crops are brought to one size."""
    pred_mask = _resize_mask(pred_mask, pred_image.shape[:2])
    gt_mask = _resize_mask(gt_mask, gt_image.shape[:2])

    pred_bbox = _bbox_from_mask(pred_mask, pad=pad)
    gt_bbox = _bbox_from_mask(gt_mask, pad=pad)
    if pred_bbox is None or gt_bbox is None:
        return float("nan"), float("nan")

    pred_crop = _crop_to_bbox(pred_image, pred_bbox)
    gt_crop = _crop_to_bbox(gt_image, gt_bbox)
    pred_mask_crop = _crop_to_bbox(pred_mask, pred_bbox)
    gt_mask_crop = _crop_to_bbox(gt_mask, gt_bbox)

    pred_crop, pred_mask_crop = _resize_image_and_mask(pred_crop, pred_mask_crop, size)
    gt_crop, gt_mask_crop = _resize_image_and_mask(gt_crop, gt_mask_crop, size)

    valid = (pred_mask_crop > 0) & (gt_mask_crop > 0)
    if int(valid.sum()) == 0:
        return float("nan"), float("nan")

    pred_lab = rgb2lab(np.clip(pred_crop, 0.0, 1.0))
    gt_lab = rgb2lab(np.clip(gt_crop, 0.0, 1.0))
    delta_e = np.sqrt(np.sum((pred_lab - gt_lab) ** 2, axis=-1))
    distance = float(delta_e[valid].mean())
    sim = float(math.exp(-distance / max(float(tau), 1e-6)))
    return distance, float(np.clip(sim, 0.0, 1.0))


@dataclass
class ReferenceView:
    path: str
    image: np.ndarray
    mask: Optional[np.ndarray] = None
    key: str = ""


class ShapeEncoder:
    def __init__(self, model_name: str, device: str):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        if device.startswith("cuda") and torch.cuda.is_available():
            self.model.to(device)

    def encode(self, image: np.ndarray) -> np.ndarray:
        pil_image = Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8))
        inputs = self.processor(images=pil_image, return_tensors="pt")
        if self.device.startswith("cuda") and self.torch.cuda.is_available():
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state[:, 0].detach().float().cpu().numpy()[0]
        return _l2_normalize(features)


class IdentityScorer:
    def __init__(self, model_name: str, device: str):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.eval()
        if device.startswith("cuda") and torch.cuda.is_available():
            self.model.to(device)
        self.image_cache: Dict[str, np.ndarray] = {}

    @staticmethod
    def _output_to_feature_vector(output: object) -> np.ndarray:
        tensor = None
        if hasattr(output, "detach"):
            tensor = output
        elif hasattr(output, "image_embeds") and getattr(output, "image_embeds") is not None:
            tensor = getattr(output, "image_embeds")
        elif hasattr(output, "pooler_output") and getattr(output, "pooler_output") is not None:
            tensor = getattr(output, "pooler_output")
        elif hasattr(output, "last_hidden_state") and getattr(output, "last_hidden_state") is not None:
            tensor = getattr(output, "last_hidden_state")[:, 0]
        elif isinstance(output, (tuple, list)) and output:
            first = output[0]
            if hasattr(first, "detach"):
                tensor = first[:, 0] if getattr(first, "ndim", 0) == 3 else first
        if tensor is None or not hasattr(tensor, "detach"):
            raise TypeError(f"Unsupported CLIP output type: {type(output)!r}")
        array = tensor.detach().float().cpu().numpy()
        if array.ndim == 3:
            array = array[:, 0, :]
        if array.ndim == 1:
            return array
        return array[0]

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        pil_image = Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8))
        inputs = self.processor(images=pil_image, return_tensors="pt")
        if self.device.startswith("cuda") and self.torch.cuda.is_available():
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            if hasattr(self.model, "get_image_features"):
                output = self.model.get_image_features(**inputs)
            else:
                output = self.model(**inputs)
            features = self._output_to_feature_vector(output)
        return _l2_normalize(features)

    def encode_reference(self, view: ReferenceView, roi_pad: int, fill_value: float) -> np.ndarray:
        cache_key = f"{view.path}|pad={roi_pad}|fill={fill_value}"
        if cache_key in self.image_cache:
            return self.image_cache[cache_key]
        mask = view.mask
        if mask is None:
            mask = _full_mask(view.image.shape[:2])
        else:
            mask = _resize_mask(mask, view.image.shape[:2])
        roi = _masked_crop_to_mask(view.image, mask, pad=roi_pad, fill_value=fill_value)
        vector = self.encode_image(roi)
        self.image_cache[cache_key] = vector
        return vector

    def best_urdf_view(self, image_roi: np.ndarray, views: Sequence[ReferenceView], roi_pad: int, fill_value: float) -> Dict[str, float | str]:
        if not views:
            return {
                "clip_score": float("nan"),
                "clip_cos": float("nan"),
                "best_ref": "",
                "num_refs": 0.0,
            }
        pred_features = self.encode_image(image_roi)
        best_score = -1.0
        best_cos = -1.0
        best_ref = ""
        for view in views:
            ref_features = self.encode_reference(view, roi_pad=roi_pad, fill_value=fill_value)
            cos = _cosine(pred_features, ref_features)
            score = 0.5 * (1.0 + cos)
            if score > best_score:
                best_score = score
                best_cos = cos
                best_ref = view.path
        return {
            "clip_score": float(np.clip(best_score, 0.0, 1.0)),
            "clip_cos": float(np.clip(best_cos, -1.0, 1.0)),
            "best_ref": best_ref,
            "num_refs": float(len(views)),
        }


@dataclass
class HandEditMetricConfig:
    roi_pad: int = 8
    contact_band_width_px: int = 12
    interaction_min_crop_size: int = 64
    shape_model_name: str = "facebook/dinov2-base"
    clip_model_name: str = "openai/clip-vit-base-patch32"
    lab_tau: float = 25.0
    lab_size: int = 224
    id_clip_weight: float = 0.5
    id_lab_weight: float = 0.5


class HandEditMetricEngine:
    def __init__(self, device: str, lpips_engine: LPIPSEngine, config: HandEditMetricConfig):
        self.lpips_engine = lpips_engine
        self.config = config
        self.shape_encoder = ShapeEncoder(config.shape_model_name, device)
        self.identity_scorer = IdentityScorer(config.clip_model_name, device)

    def compute(
        self,
        src_image: np.ndarray,
        pred_image: np.ndarray,
        gt_image: Optional[np.ndarray],
        roi_mask: np.ndarray,
        instruction: str = "",
        target_description: str = "",
        candidate_descriptions: Optional[List[str]] = None,
        object_mask: Optional[np.ndarray] = None,
        target_reference_views: Optional[Sequence[ReferenceView]] = None,
        target_reference_bank: Optional[Dict[str, np.ndarray]] = None,
        target_reference_key: str = "",
        target_name: str = "",
        replacement_scope: str = "",
        test_mask: Optional[np.ndarray] = None,
        gt_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, float | str]:
        del instruction, target_description, candidate_descriptions, target_reference_bank, target_reference_key, target_name, replacement_scope

        roi_mask = (roi_mask > 0).astype(np.uint8)
        if roi_mask.sum() <= 10:
            return {}

        output: Dict[str, float | str] = {}

        skin = _skin_mask(pred_image)
        residual_human = float(np.logical_and(skin > 0, roi_mask > 0).sum())
        roi_area = float(roi_mask.sum())
        output["Removal"] = float(np.clip(1.0 - residual_human / (roi_area + 1e-8), 0.0, 1.0))
        output["Removal_pct"] = 100.0 * float(output["Removal"])

        pred_roi = _crop_to_mask(pred_image, roi_mask, pad=self.config.roi_pad)

        output["Fidelity-Shape"] = float("nan")
        if gt_image is not None:
            gt_roi = _crop_to_mask(gt_image, roi_mask, pad=self.config.roi_pad)
            pred_features = self.shape_encoder.encode(pred_roi)
            gt_features = self.shape_encoder.encode(gt_roi)
            raw_cosine = _cosine(pred_features, gt_features)
            output["Fidelity-Shape"] = 0.5 * (1.0 + float(np.clip(raw_cosine, -1.0, 1.0)))

        # ID: CLIP against URDF views; color against GT. These two terms are kept
        # separate so the render bank does not double as a color target.
        if test_mask is None:
            if gt_mask is not None:
                test_mask = _resize_mask(gt_mask, pred_image.shape[:2])
            else:
                test_mask = _full_mask(pred_image.shape[:2])
        else:
            test_mask = _resize_mask(test_mask, pred_image.shape[:2])
        if gt_image is not None:
            if gt_mask is None:
                gt_mask = _full_mask(gt_image.shape[:2])
            else:
                gt_mask = _resize_mask(gt_mask, gt_image.shape[:2])

        pred_id_roi = _masked_crop_to_mask(pred_image, test_mask, pad=self.config.roi_pad, fill_value=0.5)
        clip_info = self.identity_scorer.best_urdf_view(
            pred_id_roi,
            list(target_reference_views or []),
            roi_pad=self.config.roi_pad,
            fill_value=0.5,
        )
        output["Fidelity-ID-clip"] = float(clip_info["clip_score"])
        output["Fidelity-ID-clip-cos"] = float(clip_info["clip_cos"])
        output["Fidelity-ID-best-ref"] = str(clip_info["best_ref"])
        output["Fidelity-ID-bank-size"] = float(clip_info["num_refs"])

        output["Fidelity-ID-lab"] = float("nan")
        output["Fidelity-ID-lab-dist"] = float("nan")
        if gt_image is not None and gt_mask is not None:
            lab_dist, lab_sim = _masked_lab_similarity(
                pred_image,
                test_mask,
                gt_image,
                gt_mask,
                pad=self.config.roi_pad,
                size=self.config.lab_size,
                tau=self.config.lab_tau,
            )
            output["Fidelity-ID-lab-dist"] = lab_dist
            output["Fidelity-ID-lab"] = lab_sim

        w_sum = max(self.config.id_clip_weight + self.config.id_lab_weight, 1e-8)
        w_clip = self.config.id_clip_weight / w_sum
        w_lab = self.config.id_lab_weight / w_sum
        clip_value = float(output["Fidelity-ID-clip"])
        lab_value = float(output["Fidelity-ID-lab"])
        if np.isfinite(clip_value) and np.isfinite(lab_value):
            output["Fidelity-ID"] = float(np.clip(w_clip * clip_value + w_lab * lab_value, 0.0, 1.0))
        elif np.isfinite(clip_value):
            output["Fidelity-ID"] = clip_value
        elif np.isfinite(lab_value):
            output["Fidelity-ID"] = lab_value
        else:
            output["Fidelity-ID"] = float("nan")

        # Backward-compatible fields, now just aliases/diagnostics for the new ID term.
        output["Fidelity-ID-margin"] = float("nan")
        output["Fidelity-ID-top1"] = float("nan")
        output["Fidelity-ID-rank-score"] = float("nan")

        shape_value = float(output["Fidelity-Shape"])
        id_value = float(output["Fidelity-ID"])
        if np.isfinite(shape_value) and np.isfinite(id_value):
            output["Fidelity"] = _harmonic_mean(shape_value, id_value)
        elif np.isfinite(id_value):
            output["Fidelity"] = id_value
        elif np.isfinite(shape_value):
            output["Fidelity"] = shape_value
        else:
            output["Fidelity"] = float("nan")
        output["Fidelity_pct"] = 100.0 * float(output["Fidelity"]) if np.isfinite(float(output["Fidelity"])) else float("nan")

        if object_mask is not None and object_mask.sum() > 0:
            interaction_mask = _dilate_mask((object_mask > 0).astype(np.uint8), self.config.contact_band_width_px)
        else:
            interaction_mask = _contact_band(roi_mask, self.config.contact_band_width_px)

        if interaction_mask.sum() > 0:
            bbox = _bbox_from_mask(interaction_mask, pad=self.config.roi_pad)
            assert bbox is not None
            src_context = _ensure_min_size(_crop_to_bbox(src_image, bbox), self.config.interaction_min_crop_size)
            pred_context = _ensure_min_size(_crop_to_bbox(pred_image, bbox), self.config.interaction_min_crop_size)
            interaction = 1.0 - float(self.lpips_engine(src_context, pred_context))
            output["Interaction"] = float(np.clip(interaction, 0.0, 1.0))
            output["Interaction_pct"] = 100.0 * float(output["Interaction"])
            output["Interaction-LPIPS"] = 1.0 - float(output["Interaction"])
        else:
            output["Interaction"] = float("nan")
            output["Interaction_pct"] = float("nan")
            output["Interaction-LPIPS"] = float("nan")

        return output
