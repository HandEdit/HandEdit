from __future__ import annotations

import base64
import json
import math
import os
import re
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageFilter


@dataclass
class VLMJudgeConfig:
    model: str = "gpt-4o"
    api_key_path: str = ""
    sleep: float = 0.3
    base_url: str = ""
    roi_crop_pad: int = 24
    contact_crop_pad: int = 32
    min_crop_size: int = 96


SC_SUBSCORES = [
    "robotness",
    "target_embodiment_match",
    "interaction_preservation",
    "scene_preservation",
]
PQ_SUBSCORES = [
    "naturalness",
    "artifact_absence",
    "local_coherence",
]
VLM_ID_SUBSCORES = [
    "correct_ref_alignment",
    "nearest_wrong_ref_alignment",
    "identity_margin",
]


def load_cache(path: str) -> Dict[str, Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return {}
    cache: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = str(row.get("id", ""))
            if sample_id:
                cache[sample_id] = row
    return cache


def append_cache(path: str, row: Dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_vlm_fields(metrics_row: Dict[str, Any], judge_row: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(metrics_row)
    mapping = {
        "SC": "VLM-SC",
        "PQ": "VLM-PQ",
        "VLM": "VLM",
        "robotness": "VLM-robotness",
        "target_embodiment_match": "VLM-target_embodiment_match",
        "interaction_preservation": "VLM-interaction_preservation",
        "scene_preservation": "VLM-scene_preservation",
        "naturalness": "VLM-naturalness",
        "artifact_absence": "VLM-artifact_absence",
        "local_coherence": "VLM-local_coherence",
        "target_choice_correct": "VLM-target_choice_correct",
        "correct_ref_alignment": "VLM-correct_ref_alignment",
        "nearest_wrong_ref_alignment": "VLM-nearest_wrong_ref_alignment",
        "identity_margin": "VLM-identity_margin",
    }
    for src_key, dst_key in mapping.items():
        if src_key in judge_row:
            merged[dst_key] = judge_row[src_key]
    passthrough = [
        "judge_mode",
        "chosen_reference",
        "vlm_has_urdf_ref",
        "vlm_num_distractors",
        "vlm_has_roi_crop",
        "vlm_has_contact_crop",
        "reasoning_sc",
        "reasoning_pq",
    ]
    for key in passthrough:
        if key in judge_row:
            merged[f"VLM-{key}" if key not in {"judge_mode", "reasoning_sc", "reasoning_pq"} else key] = judge_row[key]
    return merged




def _normalize_proxy_env() -> None:
    """Make proxy env vars compatible with httpx/OpenAI client.

    Some environments export SOCKS proxies as socks://host:port. httpx accepts
    socks5:// or socks4://, but raises ValueError for the generic socks://
    scheme. Convert it in-place before constructing the OpenAI client.
    """
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        value = os.environ.get(name)
        if value and value.lower().startswith("socks://"):
            os.environ[name] = "socks5://" + value[len("socks://"):]


def _read_api_key(path: str) -> str:
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            return handle.readline().strip()
    return ""


def _pil_to_data_url(image: Image.Image) -> str:
    image = image.convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _image_to_data_url(path: str) -> str:
    image = Image.open(path).convert("RGB")
    return _pil_to_data_url(image)


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def _to_score_1_to_5(value: Any) -> Optional[int]:
    try:
        score = int(round(float(value)))
    except Exception:
        return None
    return max(1, min(5, score))


def _to_binary(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1", "correct", "target", "a"}:
        return 1
    if text in {"false", "no", "n", "0", "incorrect"}:
        return 0
    try:
        return int(float(text) > 0.5)
    except Exception:
        return None


def _expand_box_to_min_size(
    box: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
    min_size: int,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    width, height = image_size
    crop_w = x1 - x0
    crop_h = y1 - y0
    target_w = max(crop_w, min_size)
    target_h = max(crop_h, min_size)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    x0 = int(round(cx - target_w / 2.0))
    x1 = int(round(cx + target_w / 2.0))
    y0 = int(round(cy - target_h / 2.0))
    y1 = int(round(cy + target_h / 2.0))
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y1 > height:
        y0 -= y1 - height
        y1 = height
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(width, max(x0 + 1, x1))
    y1 = min(height, max(y0 + 1, y1))
    return x0, y0, x1, y1


def _crop_by_mask(
    image_path: str,
    mask_path: str,
    pad: int = 24,
    min_crop_size: int = 96,
    dilate_px: int = 0,
) -> Optional[Image.Image]:
    if not image_path or not mask_path or not os.path.exists(image_path) or not os.path.exists(mask_path):
        return None
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, resample=Image.NEAREST)
    mask = mask.point(lambda p: 255 if p > 127 else 0)
    if dilate_px > 0:
        # MaxFilter size must be odd.
        kernel = max(3, int(dilate_px) * 2 + 1)
        if kernel % 2 == 0:
            kernel += 1
        mask = mask.filter(ImageFilter.MaxFilter(kernel))
    box = mask.getbbox()
    if box is None:
        return None
    x0, y0, x1, y1 = box
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(image.size[0], x1 + pad)
    y1 = min(image.size[1], y1 + pad)
    x0, y0, x1, y1 = _expand_box_to_min_size((x0, y0, x1, y1), image.size, min_crop_size)
    return image.crop((x0, y0, x1, y1))



def _crop_by_union_masks(
    image_path: str,
    mask_paths: Sequence[str],
    pad: int = 24,
    min_crop_size: int = 96,
    dilate_px: int = 0,
) -> Optional[Image.Image]:
    """Crop an image by the union of multiple mask files.

    Used by the VLM judge so that an empty roi_mask_path can fall back to
    human_mask_path ∪ robot_mask_path, matching the automatic metric ROI logic.
    """
    valid_paths = [path for path in mask_paths if path and os.path.exists(path)]
    if not image_path or not os.path.exists(image_path) or not valid_paths:
        return None
    image = Image.open(image_path).convert("RGB")
    merged = None
    for path in valid_paths:
        mask = Image.open(path).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, resample=Image.NEAREST)
        mask = mask.point(lambda p: 255 if p > 127 else 0)
        merged = mask if merged is None else Image.composite(Image.new("L", image.size, 255), merged, mask)
    if merged is None:
        return None
    if dilate_px > 0:
        kernel = max(3, int(dilate_px) * 2 + 1)
        if kernel % 2 == 0:
            kernel += 1
        merged = merged.filter(ImageFilter.MaxFilter(kernel))
    box = merged.getbbox()
    if box is None:
        return None
    x0, y0, x1, y1 = box
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(image.size[0], x1 + pad)
    y1 = min(image.size[1], y1 + pad)
    x0, y0, x1, y1 = _expand_box_to_min_size((x0, y0, x1, y1), image.size, min_crop_size)
    return image.crop((x0, y0, x1, y1))


def _append_image(content: List[Dict[str, Any]], label: str, image_or_path: str | Image.Image | None) -> bool:
    if image_or_path is None:
        return False
    try:
        if isinstance(image_or_path, Image.Image):
            url = _pil_to_data_url(image_or_path)
        else:
            if not image_or_path or not os.path.exists(image_or_path):
                return False
            url = _image_to_data_url(image_or_path)
    except Exception:
        return False
    content.extend([
        {"type": "text", "text": label},
        {"type": "image_url", "image_url": {"url": url}},
    ])
    return True


class OpenAIJudge:
    def __init__(self, config: VLMJudgeConfig):
        from openai import OpenAI  # type: ignore

        api_key = _read_api_key(config.api_key_path) or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set and no api_key_path was provided.")
        base_url = config.base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL") or None
        # VAPI/OpenAI-compatible endpoints are still called through the OpenAI SDK.
        # The SDK/httpx reads HTTP_PROXY/HTTPS_PROXY/ALL_PROXY by default, which can
        # break local VAPI deployments when the environment contains a SOCKS proxy but
        # socksio is not installed. For explicit OpenAI-compatible base URLs, default
        # to ignoring environment proxies unless HANDEDIT_VLM_TRUST_ENV=1 is set.
        trust_env = os.getenv("HANDEDIT_VLM_TRUST_ENV", "0").strip().lower() in {"1", "true", "yes", "y"}
        _normalize_proxy_env()
        if base_url and not trust_env:
            try:
                import httpx  # type: ignore
                self.client = OpenAI(api_key=api_key, base_url=base_url, http_client=httpx.Client(trust_env=False))
            except Exception:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = config.model
        self.sleep = float(config.sleep)
        self.roi_crop_pad = int(config.roi_crop_pad)
        self.contact_crop_pad = int(config.contact_crop_pad)
        self.min_crop_size = int(config.min_crop_size)

    @staticmethod
    def _semantic_prompt(instruction: str, has_reference: bool, num_distractors: int, has_roi_crop: bool, has_contact_crop: bool) -> str:
        reference_line = (
            "(4) the correct URDF target reference render, plus visually similar wrong target references when available."
            if has_reference
            else "(4) no URDF reference render is available; rely on the text instruction only."
        )
        crop_line = (
            "You are also given a zoomed crop of the edited hand/robot region. Use this crop for fine-grained embodiment identity, residual human-hand evidence, and local geometry."
            if has_roi_crop
            else "No ROI crop is available; judge from the full image."
        )
        contact_line = (
            "You are also given source and edited object-contact crops. Use them for interaction preservation."
            if has_contact_crop
            else "No object-contact crop is available; judge interaction preservation from the full image and ROI crop."
        )
        distractor_line = (
            f"The correct reference is labeled A. There are {num_distractors} distractor references labeled B, C, D, ... . "
            "For target embodiment match, explicitly decide whether the edited robot is closer to A than to any distractor, with particular attention to arm-link geometry, wrist shape, and hand morphology."
            if num_distractors > 0
            else "No distractor references are available. Score target embodiment match against the correct reference only."
        )
        return f"""
You are evaluating an edited image for the HandEdit benchmark.

Inputs:
(1) a source image showing a human hand or hand-arm interacting with an object;
(2) an edited image;
(3) a text instruction describing the target robot embodiment;
{reference_line}

{crop_line}
{contact_line}
{distractor_line}

Score the edited image on four 1-5 axes:
- robotness: whether the edited region is a plausible robot embodiment rather than a human hand or generic artifact.
- target_embodiment_match: whether the edited robot matches the requested target embodiment; prioritize the ROI crop and the distinctive arm-hand morphology in the reference render over the full-image background.
- interaction_preservation: whether the object state, contact relationship, and local manipulation context are preserved.
- scene_preservation: whether unrelated scene content is preserved.

Also return identity comparison fields when references are provided:
- chosen_reference: one of A/B/C/D/... or "unknown"; choose the reference that best matches the edited robot ROI.
- target_choice_correct: true iff chosen_reference is A.
- correct_ref_alignment: 1-5 alignment to reference A.
- nearest_wrong_ref_alignment: 1-5 alignment to the most similar distractor, or 1 if no distractor is available.
- identity_margin: 1-5 strength of evidence that A is closer than the nearest distractor; use 1 if a distractor is clearly closer, 3 if ambiguous, 5 if A is clearly closer.

Use 1 = failure, 2 = largely incorrect, 3 = partially correct or ambiguous, 4 = mostly correct, 5 = clearly successful.

Return JSON only:
{{
  "robotness": 1-5,
  "target_embodiment_match": 1-5,
  "interaction_preservation": 1-5,
  "scene_preservation": 1-5,
  "chosen_reference": "A/B/C/D/.../unknown",
  "target_choice_correct": true/false,
  "correct_ref_alignment": 1-5,
  "nearest_wrong_ref_alignment": 1-5,
  "identity_margin": 1-5,
  "reasoning": "brief explanation"
}}

Instruction:
{instruction}
""".strip()

    @staticmethod
    def _perceptual_prompt(has_roi_crop: bool, has_contact_crop: bool) -> str:
        roi_line = (
            "A zoomed edited ROI crop is provided; inspect it for robot-hand artifacts, residual human texture, boundary errors, malformed fingers, and local geometry."
            if has_roi_crop
            else "No ROI crop is available; inspect the full edited image."
        )
        contact_line = (
            "An edited object-contact crop is provided; local_coherence should focus especially on the edit boundary and robot-object contact area."
            if has_contact_crop
            else "No object-contact crop is available; local_coherence should focus on visible edited-region boundaries in the full image."
        )
        return f"""
You are evaluating the perceptual quality of an edited image for the HandEdit benchmark.

Score visual quality using the full edited image and all zoomed crops.
{roi_line}
{contact_line}

Score three 1-5 axes:
- naturalness: global photographic realism of the edited output.
- artifact_absence: absence of visual artifacts, blur, tearing, duplicated parts, floating geometry, or residual human-hand texture.
- local_coherence: coherence of the edited robot region, edit boundary, and robot-object contact neighborhood.

Use 1 = very poor, 2 = poor, 3 = acceptable, 4 = good, 5 = high quality.

Return JSON only:
{{
  "naturalness": 1-5,
  "artifact_absence": 1-5,
  "local_coherence": 1-5,
  "reasoning": "brief explanation"
}}
""".strip()

    def _chat(self, content: list[dict[str, Any]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=900,
        )
        return response.choices[0].message.content or ""

    def score_one(
        self,
        sample_id: str,
        src_path: str,
        pred_path: str,
        instruction: str,
        urdf_ref_path: str = "",
        roi_mask_path: str = "",
        human_mask_path: str = "",
        robot_mask_path: str = "",
        object_mask_path: str = "",
        hard_negative_ref_paths: Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        hard_negative_ref_paths = list(hard_negative_ref_paths or [])
        has_reference = bool(urdf_ref_path and os.path.exists(urdf_ref_path))

        edited_roi_crop = _crop_by_mask(
            pred_path,
            roi_mask_path,
            pad=self.roi_crop_pad,
            min_crop_size=self.min_crop_size,
            dilate_px=0,
        )
        if edited_roi_crop is None:
            edited_roi_crop = _crop_by_union_masks(
                pred_path,
                [human_mask_path, robot_mask_path],
                pad=self.roi_crop_pad,
                min_crop_size=self.min_crop_size,
                dilate_px=0,
            )
        source_contact_crop = _crop_by_mask(
            src_path,
            object_mask_path,
            pad=self.contact_crop_pad,
            min_crop_size=self.min_crop_size,
            dilate_px=max(1, self.contact_crop_pad // 4),
        )
        edited_contact_crop = _crop_by_mask(
            pred_path,
            object_mask_path,
            pad=self.contact_crop_pad,
            min_crop_size=self.min_crop_size,
            dilate_px=max(1, self.contact_crop_pad // 4),
        )
        # Fallback: if no object/contact mask is available, use the edited ROI crop as
        # the localized perceptual crop instead of judging only the full image.
        if edited_contact_crop is None and edited_roi_crop is not None:
            edited_contact_crop = edited_roi_crop
        has_roi_crop = edited_roi_crop is not None
        has_contact_crop = edited_contact_crop is not None

        valid_distractors = [path for path in hard_negative_ref_paths if path and os.path.exists(path)]
        semantic_content: List[Dict[str, Any]] = [
            {"type": "text", "text": self._semantic_prompt(instruction, has_reference, len(valid_distractors), has_roi_crop, source_contact_crop is not None and edited_contact_crop is not None)},
        ]
        _append_image(semantic_content, "Source full image:", src_path)
        _append_image(semantic_content, "Edited full image:", pred_path)
        _append_image(semantic_content, "Edited ROI crop / zoom:", edited_roi_crop)
        _append_image(semantic_content, "Source object-contact crop:", source_contact_crop)
        _append_image(semantic_content, "Edited object-contact crop:", edited_contact_crop)
        if has_reference:
            _append_image(semantic_content, "Reference A: correct URDF target render:", urdf_ref_path)
        for index, ref_path in enumerate(valid_distractors):
            label = chr(ord("B") + index)
            _append_image(semantic_content, f"Reference {label}: wrong target distractor:", ref_path)

        perceptual_content: List[Dict[str, Any]] = [
            {"type": "text", "text": self._perceptual_prompt(has_roi_crop, has_contact_crop)},
        ]
        _append_image(perceptual_content, "Edited full image:", pred_path)
        _append_image(perceptual_content, "Edited ROI crop / zoom:", edited_roi_crop)
        _append_image(perceptual_content, "Edited object-contact or boundary crop:", edited_contact_crop)

        semantic = _extract_json(self._chat(semantic_content))
        perceptual = _extract_json(self._chat(perceptual_content))

        row: Dict[str, Any] = {"id": sample_id, "judge_mode": "openai_handedit_roi_ref_distractor"}
        for key in SC_SUBSCORES:
            row[key] = _to_score_1_to_5(semantic.get(key))
        for key in VLM_ID_SUBSCORES:
            row[key] = _to_score_1_to_5(semantic.get(key))
        for key in PQ_SUBSCORES:
            row[key] = _to_score_1_to_5(perceptual.get(key))

        row["chosen_reference"] = str(semantic.get("chosen_reference", ""))
        row["target_choice_correct"] = _to_binary(semantic.get("target_choice_correct"))
        row["reasoning_sc"] = semantic.get("reasoning", "")
        row["reasoning_pq"] = perceptual.get("reasoning", "")
        row["vlm_has_urdf_ref"] = has_reference
        row["vlm_num_distractors"] = len(valid_distractors)
        row["vlm_has_roi_crop"] = bool(has_roi_crop)
        row["vlm_has_contact_crop"] = bool(has_contact_crop)

        # Make target_embodiment_match conservative when the closed-set reference
        # choice contradicts the correct target. This prevents visually plausible
        # generic robot hands from receiving a high target identity score.
        if row.get("target_choice_correct") == 0 and row.get("target_embodiment_match") is not None:
            row["target_embodiment_match"] = min(int(row["target_embodiment_match"]), 3)

        semantic_values = [row[key] for key in SC_SUBSCORES if row.get(key) is not None]
        perceptual_values = [row[key] for key in PQ_SUBSCORES if row.get(key) is not None]
        if len(semantic_values) == len(SC_SUBSCORES):
            semantic_min = min(semantic_values)
            row["SC-raw-min"] = semantic_min
            row["SC"] = (semantic_min - 1.0) / 4.0
        if len(perceptual_values) == len(PQ_SUBSCORES):
            perceptual_min = min(perceptual_values)
            row["PQ-raw-min"] = perceptual_min
            row["PQ"] = (perceptual_min - 1.0) / 4.0
        if row.get("SC") is not None and row.get("PQ") is not None:
            row["VLM"] = math.sqrt(max(float(row["SC"]), 0.0) * max(float(row["PQ"]), 0.0))

        if self.sleep > 0:
            time.sleep(self.sleep)
        return row
