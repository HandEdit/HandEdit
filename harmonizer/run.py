import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="Path to the Harmonizer repository")
    parser.add_argument("--input", default="example", help="Input folder containing composite/ and mask/")
    parser.add_argument("--weights", default="harmonizer_hand.pth")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    repo = Path(args.repo).resolve()
    input_dir = (root / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    weights = (root / args.weights).resolve() if not Path(args.weights).is_absolute() else Path(args.weights)

    command = [
        "python", "-m", "demo.image_harmonization.run",
        "--example-path", str(input_dir),
        "--pretrained", str(weights),
    ]

    env = dict(__import__("os").environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    subprocess.run(command, cwd=repo, env=env, check=True)


if __name__ == "__main__":
    main()
