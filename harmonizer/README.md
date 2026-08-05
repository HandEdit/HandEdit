# HandEdit Harmonizer

This folder contains the lightweight harmonization checkpoint used to refine HandEdit pseudo-references and a small inference wrapper for the official [Harmonizer](https://github.com/ZHKKKe/Harmonizer) implementation.

## Setup

```bash
git clone https://github.com/ZHKKKe/Harmonizer.git
cd Harmonizer
pip install -r requirements.txt
```

Prepare an input directory with matching files in `composite/` and `mask/`; results are written to `harmonized/`.

```text
example/
├── composite/
│   └── sample.jpg
└── mask/
    └── sample.png
```

Run from this repository:

```bash
python harmonizer/run.py \
  --repo /path/to/Harmonizer \
  --input /path/to/example \
  --weights harmonizer_hand.pth \
  --gpu 0
```

The mask should cover the rendered robot hand or hand-arm region. The wrapper forwards the input to the upstream image-harmonization script without changing its inference procedure.
