from __future__ import annotations

import argparse
import logging
import os

from handedit_eval.io_utils import ensure_dir
from handedit_eval.manifest import read_manifest_jsonl
from handedit_eval.runner import EvalConfig, EvalRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate HandEdit image editing results."
    )
    parser.add_argument("--manifest", required=True, help="Path to a JSONL manifest.")
    parser.add_argument("--experiment", required=True, help="Run name used for output folders.")
    parser.add_argument("--output-dir", default="runs", help="Root directory for evaluation outputs.")
    parser.add_argument("--device", default="cuda", help="Evaluation device: cuda or cpu.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )

    parser.add_argument("--disable-fid", action="store_true", help="Skip FID computation.")
    parser.add_argument(
        "--fid-roi-mode",
        default="crop",
        choices=["crop", "mask"],
        help="How ROI images are prepared for ROI-FID.",
    )
    parser.add_argument("--fid-batches", type=int, default=64, help="Batch size used for FID.")
    parser.add_argument(
        "--fid-max-items",
        type=int,
        default=0,
        help="Optional cap on the number of samples used for FID.",
    )
    parser.add_argument(
        "--fid-mode",
        default="clean",
        choices=["clean", "legacy"],
        help="clean-fid mode.",
    )

    parser.add_argument("--roi-pad", type=int, default=8, help="Padding added around ROI crops.")
    parser.add_argument(
        "--roi-dilate-px",
        type=int,
        default=0,
        help="Optional dilation applied to the ROI mask in pixels.",
    )
    parser.add_argument(
        "--roi-dilate-ratio",
        type=float,
        default=0.0,
        help="Optional dilation applied to the ROI mask as a fraction of min(H, W).",
    )

    parser.add_argument("--disable-embodied", action="store_true", help="Skip HandEdit embodied metrics.")
    parser.add_argument(
        "--contact-band-width-px",
        type=int,
        default=12,
        help="Fallback band width used for Interaction when object_mask_path is missing.",
    )
    parser.add_argument(
        "--interaction-min-crop-size",
        type=int,
        default=64,
        help="Minimum short-side size for the object/contact crop used by Interaction LPIPS.",
    )
    parser.add_argument(
        "--shape-model",
        default="facebook/dinov2-base",
        help="DINOv2 model name or local path used for Fidelity-Shape.",
    )
    parser.add_argument(
        "--clip-model",
        default="openai/clip-vit-base-patch32",
        help="CLIP model name or local path used for Fidelity-ID.",
    )
    parser.add_argument(
        "--id-clip-weight",
        type=float,
        default=0.5,
        help="Weight of max CLIP similarity to URDF renders in Fidelity-ID.",
    )
    parser.add_argument(
        "--id-lab-weight",
        type=float,
        default=0.5,
        help="Weight of masked pixel-wise Lab similarity to GT in Fidelity-ID.",
    )
    parser.add_argument(
        "--lab-tau",
        type=float,
        default=25.0,
        help="Temperature for converting mean Lab distance to similarity.",
    )
    parser.add_argument(
        "--lab-size",
        type=int,
        default=224,
        help="Square size used for masked pixel-wise Lab comparison.",
    )

    parser.add_argument(
        "--vlm-mode",
        default="off",
        choices=["off", "merge", "online"],
        help="How VLM scores are obtained.",
    )
    parser.add_argument(
        "--vlm-offline-jsonl",
        default="",
        help="Offline VLM JSONL to merge when --vlm-mode merge is used.",
    )
    parser.add_argument(
        "--vlm-model",
        default="gpt-4o",
        help="Model name for online VLM judgment.",
    )
    parser.add_argument(
        "--vlm-api-key-path",
        default="",
        help="Optional file that contains the API key for the online judge.",
    )
    parser.add_argument(
        "--vlm-base-url",
        default="",
        help="Optional OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--vlm-sleep",
        type=float,
        default=0.3,
        help="Delay between online judge calls.",
    )
    parser.add_argument(
        "--vlm-cache-jsonl",
        default="",
        help="Cache file for online judge results.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(message)s")

    if not os.path.exists(args.manifest):
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    records = read_manifest_jsonl(args.manifest)
    output_root = ensure_dir(os.path.join(args.output_dir, args.experiment))
    metrics_dir = ensure_dir(os.path.join(output_root, "metrics"))

    runner = EvalRunner(
        EvalConfig(
            device=args.device,
            roi_pad=args.roi_pad,
            roi_dilate_px=args.roi_dilate_px,
            roi_dilate_ratio=args.roi_dilate_ratio,
            compute_fid=not args.disable_fid,
            fid_roi_mode=args.fid_roi_mode,
            fid_mode=args.fid_mode,
            fid_batches=args.fid_batches,
            fid_max_items=args.fid_max_items,
            fid_tmp_root=os.path.join(metrics_dir, "_fid_tmp"),
            compute_embodied=not args.disable_embodied,
            contact_band_width_px=args.contact_band_width_px,
            interaction_min_crop_size=args.interaction_min_crop_size,
            shape_model_name=args.shape_model,
            clip_model_name=args.clip_model,
            lab_tau=args.lab_tau,
            lab_size=args.lab_size,
            id_clip_weight=args.id_clip_weight,
            id_lab_weight=args.id_lab_weight,
            vlm_mode=args.vlm_mode,
            vlm_offline_jsonl=args.vlm_offline_jsonl,
            vlm_model=args.vlm_model,
            vlm_api_key_path=args.vlm_api_key_path,
            vlm_sleep=args.vlm_sleep,
            vlm_cache_jsonl=args.vlm_cache_jsonl or os.path.join(metrics_dir, "vlm_cache.jsonl"),
            vlm_base_url=args.vlm_base_url,
        )
    )
    runner.run(records, output_root=output_root)
    logging.info("Wrote outputs to %s", metrics_dir)


if __name__ == "__main__":
    main()
