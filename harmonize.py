"""Refine composite frames with the HandEdit Harmonizer checkpoint."""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from frames import index_frames, new_output, require_same_frames


def harmonize(composite_dir, mask_dir, destination, repo, checkpoint, device=None):
    import torch
    from torchvision.transforms.functional import to_tensor

    repo = repo.resolve()
    if not (repo / "src" / "model" / "__init__.py").is_file():
        raise FileNotFoundError(f"Cannot find the Harmonizer model in {repo}")
    sys.path.insert(0, str(repo))
    try:
        from src.model import Harmonizer
    finally:
        sys.path.pop(0)

    frames = index_frames(composite_dir)
    masks = index_frames(mask_dir)
    require_same_frames(frames, masks, "Robot masks")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = Harmonizer().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True),
                          strict=True)
    model.eval()

    with new_output(destination) as output, torch.inference_mode():
        for stem, path in frames.items():
            with Image.open(path) as image, Image.open(masks[stem]) as mask_image:
                if image.size != mask_image.size:
                    raise ValueError(f"Composite and robot mask sizes differ: {stem}")
                mask = mask_image.getchannel("A") if "A" in mask_image.getbands() else mask_image.convert("L")
                mask = Image.fromarray((np.asarray(mask) > 127).astype(np.uint8) * 255)
                if mask.getbbox() is None:
                    raise ValueError(f"Empty robot mask: {stem}")
                comp = to_tensor(image.convert("RGB")).unsqueeze(0).to(device)
                mask = to_tensor(mask).unsqueeze(0).to(device)
            arguments = model.predict_arguments(comp, mask)
            result = model.restore_image(comp, mask, arguments)[-1]
            if not torch.isfinite(result).all():
                raise ValueError(f"Nonfinite Harmonizer output: {stem}")
            pixels = result[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0)
            Image.fromarray(pixels.cpu().numpy()).save(output / f"{stem}.png")
    return len(frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Composite frames")
    parser.add_argument("--masks", type=Path, required=True, help="Binary masks or RGBA robot cutouts")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--repo", type=Path, default=Path("Harmonizer"))
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/harmonizer_hand.pth"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args()
    harmonize(args.input, args.masks, args.output, args.repo, args.checkpoint, args.device)


if __name__ == "__main__":
    main()
