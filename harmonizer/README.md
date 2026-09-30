# HandEdit Harmonizer

This folder contains the lightweight harmonization checkpoint used to refine HandEdit pseudo-references and a small inference wrapper for the official [Harmonizer](https://github.com/ZHKKKe/Harmonizer) implementation.

## Benchmark references

The final harmonized outputs are the pseudo-GT references for the official Hand-only and Hand-Arm benchmarks and subsequent experiments, as described in the paper. Point the evaluation manifest's `gt_path` and `build_manifest.py --gt-root` to these outputs; the unharmonized composites are intermediate construction artifacts.

## Setup

```bash
git clone https://github.com/ZHKKKe/Harmonizer.git
cd Harmonizer
pip install -r requirements.txt
```

Place composite images in `composite/` and their corresponding masks in `mask/`, using matching basenames. The outputs are saved to `harmonized/`.

```text
example/
├── composite/
│   └── sample.jpg
└── mask/
    └── sample.png
```

From the HandEdit repository root, run:

```bash
python harmonizer/run.py \
  --repo /path/to/Harmonizer \
  --input /path/to/example \
  --weights harmonizer_hand.pth \
  --gpu 0
```

The mask should cover the rendered robot hand or hand-arm region. The wrapper forwards the input to the upstream image-harmonization script without changing its inference procedure.
