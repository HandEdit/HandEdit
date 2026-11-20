"""Track one hand across a clip, as binary masks or RGBA cutouts."""

import argparse
import os
import shutil
import sys
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from frames import index_frames, new_output


def track(predictor, directory, output, prompt, cutouts=False, prompt_frame=0):
    frames = list(index_frames(directory).values())
    with TemporaryDirectory() as temp, new_output(output) as dest:
        resource = Path(temp)
        # SAM reads directory entries itself. Number the inputs so its indices
        # always refer to our source frames, including names like color_2.jpg.
        for i, path in enumerate(frames):
            alias = resource / f"{i:08d}{path.suffix.lower()}"
            try:
                os.link(path, alias)
            except OSError:
                shutil.copyfile(path, alias)
        _track(predictor, resource, frames, dest, prompt, cutouts, min(prompt_frame, len(frames) - 1))


def _track(predictor, resource, frames, output, prompt, cutouts, prompt_frame):
    request = predictor.handle_request
    session = request({"type": "start_session", "resource_path": str(resource)})["session_id"]
    try:
        response = request({"type": "add_prompt", "session_id": session,
                            "frame_index": prompt_frame, "text": prompt})
        tracks = response["outputs"]["out_obj_ids"]
        if len(tracks) != 1:
            raise ValueError(f"Expected one {prompt!r} in frame {frames[prompt_frame].stem}; found {len(tracks)}")
        track_id = int(tracks[0])
        seen = set()
        for result in predictor.handle_stream_request({
            "type": "propagate_in_video", "session_id": session, "propagation_direction": "forward",
            "start_frame_index": 0,
        }):
            index, objects = result["frame_index"], result["outputs"]
            if not 0 <= index < len(frames):
                raise ValueError(f"SAM 3 returned an unknown frame index: {index}")
            ids = np.asarray(objects["out_obj_ids"]).tolist()
            if track_id not in ids:
                continue
            mask = np.asarray(objects["out_binary_masks"][ids.index(track_id)])
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]
            if mask.ndim != 2 or not np.any(mask):
                continue
            source = frames[index]
            with Image.open(source) as frame:
                alpha = Image.fromarray((mask > 0).astype(np.uint8) * 255)
                alpha = alpha.resize(frame.size, Image.Resampling.NEAREST)
                image = alpha
                if cutouts:
                    image = frame.convert("RGBA")
                    image.putalpha(alpha)
                image.save(output / f"{source.stem}.png")
            seen.add(index)
        if len(seen) != len(frames):
            missing = [frame.stem for i, frame in enumerate(frames) if i not in seen]
            raise ValueError(f"Missing {prompt!r} masks for frames {missing[:5]}")
    finally:
        request({"type": "close_session", "session_id": session})


@contextmanager
def sam_predictor(checkpoint):
    import torch
    from sam3.model_builder import build_sam3_video_predictor

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError("SAM 3 video segmentation requires a CUDA GPU")
    vocabulary = files("sam3").joinpath("assets", "bpe_simple_vocab_16e6.txt.gz")
    predictor = build_sam3_video_predictor(checkpoint_path=str(checkpoint.resolve()),
                                           bpe_path=str(vocabulary), gpus_to_use=[0])
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            yield predictor
    finally:
        predictor.shutdown()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", choices=("human", "robot"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "weights" / "sam3.pt")
    args = parser.parse_args()
    with sam_predictor(args.checkpoint) as predictor:
        if args.subject == "human":
            track(predictor, args.input, args.output, "hand")
        else:
            track(predictor, args.input, args.output, "robot hand", cutouts=True, prompt_frame=30)
