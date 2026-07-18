from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_dilation
from tqdm import tqdm

from .bank import canonicalize_target_name, enrich_record, infer_group
from .io_utils import (
    apply_mask_rgb,
    bbox_from_mask,
    crop,
    ensure_dir,
    invert_mask,
    load_image_rgb,
    load_mask,
    resize_mask_to,
    resize_to_match,
    union_masks,
)
from .metrics.fid import compute_fid
from .metrics.generic import LPIPSEngine, psnr, ssim_rgb
from .metrics.handedit import HandEditMetricConfig, HandEditMetricEngine, ReferenceView
from .metrics.vlm import VLMJudgeConfig, OpenAIJudge, append_cache, load_cache, merge_vlm_fields
from .summary import jsonl_to_dataframe, save_leaderboard, summarize_dataframe

LOGGER = logging.getLogger(__name__)

GENERIC_METRICS = [
    "Full-PSNR", "Full-SSIM", "Full-LPIPS",
    "ROI-PSNR", "ROI-SSIM", "ROI-LPIPS",
    "BG-PSNR", "BG-SSIM", "BG-LPIPS",
]

VLM_METRICS = [
    "VLM-SC", "VLM-PQ", "VLM",
    "VLM-robotness", "VLM-target_embodiment_match", "VLM-interaction_preservation", "VLM-scene_preservation",
    "VLM-naturalness", "VLM-artifact_absence", "VLM-local_coherence",
    "VLM-target_choice_correct", "VLM-correct_ref_alignment",
    "VLM-nearest_wrong_ref_alignment", "VLM-identity_margin",
]

EMBODIED_METRICS = [
    "Removal", "Removal_pct",
    "Fidelity-Shape",
    "Fidelity-ID", "Fidelity-ID-clip", "Fidelity-ID-clip-cos",
    "Fidelity-ID-lab", "Fidelity-ID-lab-dist",
    "Fidelity-ID-bank-size", "Fidelity-ID-best-ref",
    "Fidelity-ID-margin", "Fidelity-ID-top1", "Fidelity-ID-rank-score",
    "Fidelity", "Fidelity_pct",
    "Interaction", "Interaction_pct", "Interaction-LPIPS",
]


def _safe_link_or_copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    try:
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _save_png(image: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)).save(path)


@dataclass
class EvalConfig:
    device: str = "cuda"
    roi_pad: int = 8
    roi_dilate_px: int = 0
    roi_dilate_ratio: float = 0.0
    compute_fid: bool = True
    fid_roi_mode: str = "crop"
    fid_mode: str = "clean"
    fid_batches: int = 64
    fid_max_items: int = 0
    fid_tmp_root: str = ""
    compute_embodied: bool = True
    contact_band_width_px: int = 12
    interaction_min_crop_size: int = 64
    shape_model_name: str = "facebook/dinov2-base"
    clip_model_name: str = "openai/clip-vit-base-patch32"
    lab_tau: float = 25.0
    lab_size: int = 224
    id_clip_weight: float = 0.5
    id_lab_weight: float = 0.5
    vlm_mode: str = "off"
    vlm_offline_jsonl: str = ""
    vlm_model: str = "gpt-4o"
    vlm_api_key_path: str = ""
    vlm_workers: int = 1
    vlm_sleep: float = 0.3
    vlm_cache_jsonl: str = ""
    vlm_base_url: str = ""


class EvalRunner:
    def __init__(self, config: EvalConfig):
        self.config = config

    def _dilate_mask(self, mask: np.ndarray | None, hw: Tuple[int, int]) -> np.ndarray | None:
        if mask is None:
            return None
        height, width = hw
        radius = int(round(min(height, width) * self.config.roi_dilate_ratio)) if self.config.roi_dilate_ratio > 0 else int(self.config.roi_dilate_px)
        if radius <= 0:
            return mask
        yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        structure = (xx * xx + yy * yy) <= radius * radius
        return binary_dilation(mask.astype(bool), structure=structure).astype(np.uint8)

    def _load_mask_if_present(self, path: str, hw: Tuple[int, int] | None = None) -> np.ndarray | None:
        if not path or not os.path.exists(path):
            return None
        mask = load_mask(path)
        if hw is not None:
            mask = resize_mask_to(hw, mask)
        return mask

    def _resolve_roi_mask(self, sample: Dict[str, Any], hw: Tuple[int, int]) -> np.ndarray | None:
        # Keep the edit ROI definition fixed across runs. We ignore roi_mask_path
        # here on purpose; it is too easy for that mask to mean different things
        # in different manifests. The public ROI is human ∪ robot.
        human_mask = self._load_mask_if_present(str(sample.get("human_mask_path", "")), hw)
        robot_mask = self._load_mask_if_present(str(sample.get("robot_mask_path", "")), hw)
        merged = union_masks(human_mask, robot_mask)
        if merged is None:
            return None
        return self._dilate_mask(merged, hw)

    def _resolve_background_mask(self, sample: Dict[str, Any], roi_mask: np.ndarray | None, hw: Tuple[int, int]) -> np.ndarray | None:
        bg_mask = self._load_mask_if_present(str(sample.get("bg_mask_path", "")), hw)
        if bg_mask is not None:
            return bg_mask
        if roi_mask is None:
            return None
        return invert_mask(roi_mask)

    @staticmethod
    def _target_reference_key(sample: Dict[str, Any]) -> str:
        target_name = canonicalize_target_name(str(sample.get("target_name", "") or sample.get("robot_name", "")))
        replacement_scope = str(sample.get("replacement_scope", "") or sample.get("scope", ""))
        group = infer_group(target_name, replacement_scope) or replacement_scope or "unknown"
        return f"{group}:{target_name}" if target_name else ""


    def _load_reference_view(self, key: str, image_path: str, mask_path: Optional[str] = None) -> Optional[ReferenceView]:
        if not image_path or not os.path.exists(image_path):
            return None
        try:
            image = load_image_rgb(image_path)
            mask = None
            if mask_path and os.path.exists(mask_path):
                mask = resize_mask_to(image.shape[:2], load_mask(mask_path))
            return ReferenceView(path=image_path, image=image, mask=mask, key=key)
        except Exception as exc:
            LOGGER.warning("Failed to load URDF view %s: %s", image_path, exc)
            return None

    def _reference_views_from_paths(self, key: str, paths: List[str], masks: List[str]) -> List[ReferenceView]:
        views: List[ReferenceView] = []
        for idx, path in enumerate(paths):
            mask_path = masks[idx] if idx < len(masks) else ""
            view = self._load_reference_view(key, path, mask_path)
            if view is not None:
                views.append(view)
        return views

    @staticmethod
    def _path_list(value: Any) -> List[str]:
        """Normalize a manifest path field to a list of strings.

        Manifest entries may store URDF references as a list, a single string,
        or occasionally as a JSON-encoded list. Missing values become an empty
        list.
        """
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value if v]
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return []
            if value.startswith("["):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return [str(v) for v in parsed if v]
                except Exception:
                    pass
            return [value]
        return [str(value)]

    def _sample_urdf_paths(self, sample: Dict[str, Any]) -> List[str]:
        paths = self._path_list(sample.get("urdf_ref_paths"))
        if paths:
            return paths
        return self._path_list(sample.get("urdf_ref_path"))

    def _sample_urdf_masks(self, sample: Dict[str, Any]) -> List[str]:
        paths = self._path_list(sample.get("urdf_mask_paths"))
        if paths:
            return paths
        return self._path_list(sample.get("urdf_mask_path"))

    def _build_target_reference_bank(self, samples: List[Dict[str, Any]]) -> Dict[str, List[ReferenceView]]:
        """Build the small per-target render bank from manifest entries only."""
        bank: Dict[str, List[ReferenceView]] = {}
        seen: set[tuple[str, str]] = set()
        for sample in samples:
            key = self._target_reference_key(sample)
            for view in self._reference_views_from_paths(
                key, self._sample_urdf_paths(sample), self._sample_urdf_masks(sample)
            ):
                if key and (key, view.path) not in seen:
                    bank.setdefault(key, []).append(view)
                    seen.add((key, view.path))
        return bank

    def _reference_views_for_sample(
        self,
        sample: Dict[str, Any],
        bank: Dict[str, List[ReferenceView]],
    ) -> List[ReferenceView]:
        key = self._target_reference_key(sample)
        explicit = self._reference_views_from_paths(key, self._sample_urdf_paths(sample), self._sample_urdf_masks(sample))
        return explicit if explicit else list(bank.get(key, []))

    def _build_target_reference_path_bank(self, samples: List[Dict[str, Any]], reference_bank: Dict[str, List[ReferenceView]]) -> Dict[str, str]:
        bank: Dict[str, str] = {}
        for key, views in reference_bank.items():
            if views:
                bank[key] = views[0].path
        for sample in samples:
            key = self._target_reference_key(sample)
            paths = self._sample_urdf_paths(sample)
            if key and paths and key not in bank and os.path.exists(paths[0]):
                bank[key] = paths[0]
        return bank

    def _select_vlm_hard_negative_refs(
        self,
        sample: Dict[str, Any],
        path_bank: Dict[str, str],
        max_items: int = 3,
    ) -> List[str]:
        """Select same-track URDF reference distractors for the VLM judge.

        We intentionally keep this deterministic and manifest-free. If no same-track
        distractors are available, it falls back to any other target reference.
        """
        current_key = self._target_reference_key(sample)
        if not current_key:
            return []
        current_group = current_key.split(":", 1)[0]
        same_group = [
            (key, path)
            for key, path in sorted(path_bank.items())
            if key != current_key and key.split(":", 1)[0] == current_group and path and os.path.exists(path)
        ]
        if len(same_group) < max_items:
            same_group.extend([
                (key, path)
                for key, path in sorted(path_bank.items())
                if key != current_key and key.split(":", 1)[0] != current_group and path and os.path.exists(path)
            ])
        return [path for _, path in same_group[:max_items]]


    def run(self, samples: List[Dict[str, Any]], output_root: str) -> None:
        output_root = ensure_dir(output_root)
        metrics_dir = ensure_dir(os.path.join(output_root, "metrics"))
        per_sample_jsonl = os.path.join(metrics_dir, "per_sample.jsonl")

        samples = [enrich_record(sample) for sample in samples]
        target_reference_bank = self._build_target_reference_bank(samples)
        target_reference_path_bank = self._build_target_reference_path_bank(samples, target_reference_bank)
        LOGGER.info("Loaded %d URDF target groups / %d manifest views for Fidelity-ID.", len(target_reference_bank), sum(len(v) for v in target_reference_bank.values()))
        LOGGER.info("Loaded %d target reference paths for VLM distractors.", len(target_reference_path_bank))
        lpips_engine = LPIPSEngine(device=self.config.device)
        handedit_engine = HandEditMetricEngine(
            device=self.config.device,
            lpips_engine=lpips_engine,
            config=HandEditMetricConfig(
                roi_pad=self.config.roi_pad,
                contact_band_width_px=self.config.contact_band_width_px,
                interaction_min_crop_size=self.config.interaction_min_crop_size,
                shape_model_name=self.config.shape_model_name,
                clip_model_name=self.config.clip_model_name,
                lab_tau=self.config.lab_tau,
                lab_size=self.config.lab_size,
                id_clip_weight=self.config.id_clip_weight,
                id_lab_weight=self.config.id_lab_weight,
            ),
        )

        vlm_cache = {}
        offline_vlm = {}
        judge = None
        if self.config.vlm_mode == "merge":
            if not self.config.vlm_offline_jsonl or not os.path.exists(self.config.vlm_offline_jsonl):
                raise FileNotFoundError(f"Offline VLM JSONL not found: {self.config.vlm_offline_jsonl}")
            offline_vlm = load_cache(self.config.vlm_offline_jsonl)
        elif self.config.vlm_mode == "online":
            judge = OpenAIJudge(
                VLMJudgeConfig(
                    model=self.config.vlm_model,
                    api_key_path=self.config.vlm_api_key_path,
                    sleep=self.config.vlm_sleep,
                    base_url=self.config.vlm_base_url,
                    min_crop_size=self.config.interaction_min_crop_size,
                )
            )
            if self.config.vlm_cache_jsonl:
                vlm_cache = load_cache(self.config.vlm_cache_jsonl)

        with open(per_sample_jsonl, "w", encoding="utf-8") as handle:
            for sample in tqdm(samples, desc="Evaluating"):
                row = self._evaluate_one(sample, lpips_engine, handedit_engine, target_reference_bank)
                sample_id = str(sample.get("id", ""))
                if self.config.vlm_mode == "merge" and sample_id in offline_vlm:
                    row = merge_vlm_fields(row, offline_vlm[sample_id])
                elif self.config.vlm_mode == "online" and judge is not None:
                    cached = vlm_cache.get(sample_id)
                    if cached is None:
                        try:
                            cached = judge.score_one(
                                sample_id=sample_id,
                                src_path=str(sample.get("src_path", "")),
                                pred_path=str(sample.get("pred_path", "")),
                                instruction=str(sample.get("instruction", "")),
                                urdf_ref_path=(self._sample_urdf_paths(sample)[0] if self._sample_urdf_paths(sample) else ""),
                                roi_mask_path="",
                                human_mask_path=str(sample.get("human_mask_path", "")),
                                robot_mask_path=str(sample.get("robot_mask_path", "")),
                                object_mask_path=str(sample.get("object_mask_path", "") or sample.get("contact_mask_path", "")),
                                hard_negative_ref_paths=self._select_vlm_hard_negative_refs(sample, target_reference_path_bank),
                            )
                            if self.config.vlm_cache_jsonl:
                                append_cache(self.config.vlm_cache_jsonl, cached)
                        except Exception as exc:
                            cached = {"id": sample_id, "judge_mode": "error", "error": repr(exc)}
                    row = merge_vlm_fields(row, cached)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        dataframe = jsonl_to_dataframe(per_sample_jsonl)
        dataframe.to_csv(os.path.join(metrics_dir, "per_sample.csv"), index=False)

        fid_results = self._compute_fid(samples, metrics_dir) if self.config.compute_fid else {}
        summary = self._make_summary(dataframe, fid_results, exp_name=os.path.basename(output_root))
        with open(os.path.join(metrics_dir, "summary.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

        leaderboard_rows = [row for key, row in summary.items() if key != "_meta"]
        leaderboard_df = pd.DataFrame(leaderboard_rows)
        sort_key = "VLM_mean" if "VLM_mean" in leaderboard_df.columns else ("Fidelity_mean" if "Fidelity_mean" in leaderboard_df.columns else None)
        save_leaderboard(
            leaderboard_df,
            os.path.join(metrics_dir, "leaderboard.csv"),
            os.path.join(metrics_dir, "leaderboard.md"),
            sort_key=sort_key,
            ascending=False,
        )

    def _evaluate_one(
        self,
        sample: Dict[str, Any],
        lpips_engine: LPIPSEngine,
        handedit_engine: HandEditMetricEngine,
        target_reference_bank: Dict[str, List[ReferenceView]],
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "id": sample.get("id", ""),
            "dataset": sample.get("dataset", ""),
            "task": sample.get("task", ""),
            "instruction": sample.get("instruction", ""),
            "src_path": sample.get("src_path", ""),
            "pred_path": sample.get("pred_path", ""),
            "gt_path": sample.get("gt_path", ""),
            "test_mask_path": sample.get("test_mask_path", "") or sample.get("pred_mask_path", "") or self._first_path(sample.get("test_mask_paths")),
            "gt_mask_path": sample.get("gt_mask_path", "") or self._first_path(sample.get("gt_mask_paths")),
            "urdf_ref_paths": sample.get("urdf_ref_paths", sample.get("urdf_ref_path", "")),
            "urdf_mask_paths": sample.get("urdf_mask_paths", sample.get("urdf_mask_path", "")),
            "roi_mask_path": sample.get("roi_mask_path", ""),
            "human_mask_path": sample.get("human_mask_path", ""),
            "robot_mask_path": sample.get("robot_mask_path", ""),
            "bg_mask_path": sample.get("bg_mask_path", ""),
            "object_mask_path": sample.get("object_mask_path", "") or sample.get("contact_mask_path", ""),
            "urdf_ref_path": sample.get("urdf_ref_path", ""),
            "target_name": sample.get("target_name", ""),
            "replacement_scope": sample.get("replacement_scope", "") or sample.get("scope", ""),
        }
        for metric in GENERIC_METRICS + VLM_METRICS + EMBODIED_METRICS:
            result.setdefault(metric, float("nan"))

        pred_path = str(sample.get("pred_path", ""))
        src_path = str(sample.get("src_path", ""))
        gt_path = str(sample.get("gt_path", ""))
        if not pred_path or not os.path.exists(pred_path):
            result["error"] = f"Missing pred_path: {pred_path}"
            return result

        pred_image = load_image_rgb(pred_path)
        src_image = load_image_rgb(src_path) if src_path and os.path.exists(src_path) else None
        gt_image = load_image_rgb(gt_path) if gt_path and os.path.exists(gt_path) else None
        roi_mask = None

        if src_image is not None:
            pred_image, src_image = resize_to_match(pred_image, src_image)
            roi_mask = self._resolve_roi_mask(sample, src_image.shape[:2])

        if gt_image is not None:
            pred_image, gt_image = resize_to_match(pred_image, gt_image)
            if src_image is not None:
                src_image, gt_image = resize_to_match(src_image, gt_image)
                roi_mask = self._resolve_roi_mask(sample, gt_image.shape[:2])
            else:
                roi_mask = self._resolve_roi_mask(sample, gt_image.shape[:2])

        if roi_mask is None:
            roi_mask = np.ones(pred_image.shape[:2], dtype=np.uint8)

        if self.config.compute_embodied and src_image is not None:
            object_mask = None
            object_mask_path = str(sample.get("object_mask_path", "") or sample.get("contact_mask_path", ""))
            if object_mask_path and os.path.exists(object_mask_path):
                object_mask = load_mask(object_mask_path)
                object_mask = resize_mask_to(src_image.shape[:2], object_mask)
            test_mask_path = str(sample.get("test_mask_path", "") or sample.get("pred_mask_path", "") or self._first_path(sample.get("test_mask_paths")))
            gt_mask_path = str(sample.get("gt_mask_path", "") or self._first_path(sample.get("gt_mask_paths")))
            test_mask = self._load_mask_if_present(test_mask_path, pred_image.shape[:2])
            gt_mask = self._load_mask_if_present(gt_mask_path, gt_image.shape[:2]) if gt_image is not None else None
            reference_views = self._reference_views_for_sample(sample, target_reference_bank)

            embodied = handedit_engine.compute(
                src_image=src_image,
                pred_image=pred_image,
                gt_image=gt_image,
                roi_mask=roi_mask,
                instruction=str(sample.get("instruction", "")),
                target_description=str(sample.get("target_description", "") or sample.get("embodiment_description", "")),
                candidate_descriptions=list(sample.get("candidate_descriptions", []) or []),
                object_mask=object_mask,
                target_reference_views=reference_views,
                target_name=str(sample.get("target_name", "") or sample.get("robot_name", "")),
                replacement_scope=str(sample.get("replacement_scope", "") or sample.get("scope", "")),
                test_mask=test_mask,
                gt_mask=gt_mask,
            )
            result.update(embodied)

        if gt_image is None:
            return result

        pred_image, gt_image = resize_to_match(pred_image, gt_image)
        bg_mask = self._resolve_background_mask(sample, roi_mask, gt_image.shape[:2])

        result["Full-PSNR"] = psnr(pred_image, gt_image, mask=None)
        result["Full-SSIM"] = ssim_rgb(pred_image, gt_image)
        result["Full-LPIPS"] = lpips_engine(pred_image, gt_image)

        if roi_mask is not None:
            bbox = bbox_from_mask(roi_mask, pad=self.config.roi_pad)
            if bbox is not None:
                pred_roi = crop(pred_image, bbox)
                gt_roi = crop(gt_image, bbox)
                roi_crop_mask = crop(roi_mask, bbox)
                result["ROI-PSNR"] = psnr(pred_roi, gt_roi, mask=roi_crop_mask)
                result["ROI-SSIM"] = ssim_rgb(pred_roi, gt_roi)
                result["ROI-LPIPS"] = lpips_engine(pred_roi, gt_roi)

        if bg_mask is not None:
            result["BG-PSNR"] = psnr(apply_mask_rgb(pred_image, bg_mask), apply_mask_rgb(gt_image, bg_mask), mask=bg_mask)
            result["BG-SSIM"] = ssim_rgb(apply_mask_rgb(pred_image, bg_mask), apply_mask_rgb(gt_image, bg_mask))
            result["BG-LPIPS"] = lpips_engine(apply_mask_rgb(pred_image, bg_mask), apply_mask_rgb(gt_image, bg_mask))

        return result

    def _compute_fid(self, samples: List[Dict[str, Any]], metrics_dir: str) -> Dict[str, float]:
        temp_root = self.config.fid_tmp_root or ensure_dir(os.path.join(metrics_dir, "_fid_tmp"))
        directories = {
            "full_real": ensure_dir(os.path.join(temp_root, "full_real")),
            "full_gen": ensure_dir(os.path.join(temp_root, "full_gen")),
            "roi_real": ensure_dir(os.path.join(temp_root, "roi_real")),
            "roi_gen": ensure_dir(os.path.join(temp_root, "roi_gen")),
        }
        for directory in directories.values():
            for entry in os.listdir(directory):
                path = os.path.join(directory, entry)
                try:
                    if os.path.islink(path) or os.path.isfile(path):
                        os.remove(path)
                    elif os.path.isdir(path):
                        shutil.rmtree(path)
                except Exception:
                    LOGGER.warning("Could not clean temporary FID file: %s", path)

        added = 0
        for sample in samples:
            pred_path = str(sample.get("pred_path", ""))
            gt_path = str(sample.get("gt_path", ""))
            sample_id = str(sample.get("id", ""))
            if not pred_path or not gt_path or not os.path.exists(pred_path) or not os.path.exists(gt_path):
                continue
            _safe_link_or_copy(gt_path, os.path.join(directories["full_real"], f"{sample_id}.png"))
            _safe_link_or_copy(pred_path, os.path.join(directories["full_gen"], f"{sample_id}.png"))

            pred_image = load_image_rgb(pred_path)
            gt_image = load_image_rgb(gt_path)
            pred_image, gt_image = resize_to_match(pred_image, gt_image)
            roi_mask = self._resolve_roi_mask(sample, gt_image.shape[:2])
            if roi_mask is not None:
                if self.config.fid_roi_mode == "mask":
                    _save_png(apply_mask_rgb(gt_image, roi_mask), os.path.join(directories["roi_real"], f"{sample_id}.png"))
                    _save_png(apply_mask_rgb(pred_image, roi_mask), os.path.join(directories["roi_gen"], f"{sample_id}.png"))
                else:
                    bbox = bbox_from_mask(roi_mask, pad=self.config.roi_pad)
                    if bbox is not None:
                        _save_png(crop(gt_image, bbox), os.path.join(directories["roi_real"], f"{sample_id}.png"))
                        _save_png(crop(pred_image, bbox), os.path.join(directories["roi_gen"], f"{sample_id}.png"))

            added += 1
            if self.config.fid_max_items and added >= self.config.fid_max_items:
                break

        result: Dict[str, float] = {}
        try:
            result["Full-FID"] = compute_fid(directories["full_real"], directories["full_gen"], device=self.config.device, batch_size=self.config.fid_batches, mode=self.config.fid_mode) if len(os.listdir(directories["full_real"])) >= 2 else float("nan")
        except Exception as exc:
            result["Full-FID"] = float("nan")
            result["Full-FID-error"] = repr(exc)
        try:
            result["ROI-FID"] = compute_fid(directories["roi_real"], directories["roi_gen"], device=self.config.device, batch_size=self.config.fid_batches, mode=self.config.fid_mode) if len(os.listdir(directories["roi_real"])) >= 2 else float("nan")
        except Exception as exc:
            result["ROI-FID"] = float("nan")
            result["ROI-FID-error"] = repr(exc)
        return result

    def _make_summary(self, df: pd.DataFrame, fid_results: Dict[str, float], exp_name: str) -> Dict[str, Any]:
        metric_columns = [column for column in GENERIC_METRICS + VLM_METRICS + EMBODIED_METRICS if column in df.columns]
        summary_df = summarize_dataframe(df, group_by=["dataset", "task"], metric_columns=metric_columns)
        for key, value in fid_results.items():
            summary_df[key] = value
        summary_df["exp_name"] = exp_name

        result: Dict[str, Any] = {"_meta": {"exp_name": exp_name, "group_by": ["dataset", "task"]}}
        for _, row in summary_df.iterrows():
            result[f"{row.get('dataset', '')}/{row.get('task', '')}"] = row.to_dict()
        return result
