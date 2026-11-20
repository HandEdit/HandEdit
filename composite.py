"""Composite aligned robot cutouts over inpainted camera frames."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from frames import index_frames, new_output, require_same_frames


def composite(foreground_dir, background_dir, destination, fg_blur_strength=3,
              erosion_strength=3, blur_strength=12):
    if min(fg_blur_strength, erosion_strength, blur_strength) < 0:
        raise ValueError("Blur and erosion strengths must be nonnegative")
    foregrounds = index_frames(foreground_dir)
    backgrounds = index_frames(background_dir)
    require_same_frames(backgrounds, foregrounds, "Robot cutouts")
    feather_size = blur_strength + (blur_strength % 2 == 0)

    with new_output(destination) as output:
        for stem, path in foregrounds.items():
            with Image.open(path) as foreground, Image.open(backgrounds[stem]) as background:
                if "A" not in foreground.getbands():
                    raise ValueError(f"Cutout has no alpha channel: {path.name}")
                if foreground.size != background.size:
                    raise ValueError(f"Cutout and background sizes differ: {stem}")
                color = np.asarray(foreground.convert("RGB"))
                if fg_blur_strength:
                    size = 2 * fg_blur_strength - 1
                    color = cv2.GaussianBlur(color, (size, size), 0)
                alpha = np.asarray(foreground.getchannel("A"))
                if erosion_strength:
                    alpha = cv2.erode(alpha, np.ones((3, 3), dtype=np.uint8),
                                      iterations=erosion_strength)
                if not np.any(alpha):
                    raise ValueError(f"Empty cutout after erosion: {path.name}")
                if feather_size > 1:
                    alpha = cv2.GaussianBlur(alpha, (feather_size, feather_size), 0)
                result = background.convert("RGB")
                result.paste(Image.fromarray(color), mask=Image.fromarray(alpha))
                result.save(output / f"{stem}.png")
    return len(foregrounds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fg_dir", type=Path, required=True, help="RGBA robot cutouts")
    parser.add_argument("--bg_dir", type=Path, required=True, help="Inpainted RGB frames")
    parser.add_argument("--output_dir", type=Path, required=True, help="New output directory")
    parser.add_argument("--fg_blur_strength", type=int, default=3)
    parser.add_argument("--erosion_strength", type=int, default=3)
    parser.add_argument("--blur_strength", type=int, default=12)
    args = parser.parse_args()
    composite(args.fg_dir, args.bg_dir, args.output_dir, args.fg_blur_strength,
              args.erosion_strength, args.blur_strength)


if __name__ == "__main__":
    main()
