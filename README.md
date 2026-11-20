# HandEdit Preprocessing

[Project](https://handedit.github.io/) | [Dataset](https://huggingface.co/datasets/HandEdit/HandEdit) | [Models](https://huggingface.co/HandEdit/HandEdit-LoRA)

Data preprocessing for HandEdit, including our modified SAM 3, ProPainter and dex-retargeting code.

```text
.
|-- dex-retargeting-main/  # Hand motion retargeting and rendering
|-- sam3-main/             # Human hand masks and robot cutouts
|-- ProPainter-main/       # Background inpainting
|-- composite.py           # Composition and blending
|-- harmonize.py           # Image harmonization
`-- frames.py              # Shared frame I/O
```

The stage scripts share [frames.py](frames.py) for frame indexing, filename matching and output-directory handling.

## Installation

Run the commands below from the repository root.

The two environments keep dex-retargeting's NumPy requirement (`>=2`) separate from SAM 3's (`<2`).

### Retargeting

`handedit-retargeting` is used for motion retargeting and rendering.
The DexYCB example uses the Ability Hand URDF and meshes from [dex-urdf](https://github.com/dexsuite/dex-urdf).
See [offline retargeting](dex-retargeting-main/README.md#offline-retargeting-configurations) for other robot configurations.

```bash
git clone https://github.com/dexsuite/dex-urdf.git dex-retargeting-main/assets

conda create -n handedit-retargeting python=3.10 -y
conda activate handedit-retargeting
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -e ./dex-retargeting-main sapien==3.0.1 "setuptools<82"
```

The Ability Hand example expects this layout. Keep the mesh directories alongside the URDF.

```text
dex-retargeting-main/assets/robots/hands/ability_hand/
|-- ability_hand_right.urdf
`-- meshes/
    |-- visual/
    `-- collision/
```

### Image Synthesis

`handedit-synthesis` is used for segmentation, inpainting, compositing and harmonization.
Clone Harmonizer into `Harmonizer/` so that `harmonize.py` can import `Harmonizer/src/model/`.
The environment below includes its inference dependencies.

```bash
git clone https://github.com/ZHKKKe/Harmonizer.git Harmonizer

conda create -n handedit-synthesis python=3.12 -y
conda activate handedit-synthesis
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt -r ProPainter-main/requirements.txt \
  -e ./sam3-main "setuptools<82" kornia==0.8.2
```

### Checkpoints

- [SAM 3](https://huggingface.co/facebook/sam3): `weights/sam3.pt`.
- [HandEdit Harmonizer](https://github.com/HandEdit/HandEdit/blob/main/harmonizer/harmonizer_hand.pth): `weights/harmonizer_hand.pth`.
- [ProPainter](https://github.com/sczhou/ProPainter/releases/tag/v0.1.0): weights are downloaded to `ProPainter-main/weights/` on first use.

## DexYCB Example

We use one camera clip with a right Ability Hand. Download the sequences, calibration files and object models from [DexYCB](https://dex-ycb.github.io/) and keep the original layout:

```text
data/dex-ycb/
|-- calibration/
|-- models/
`-- SUBJECT/
    `-- CAPTURE/
        |-- meta.yml
        `-- CAMERA/
            |-- color_000041.jpg
            |-- labels_000041.npz
            `-- ...
```

Replace `SUBJECT`, `CAPTURE` and `CAMERA` with your sequence. The example requires `mano_sides: [right]` in `meta.yml` and a continuous clip with valid hand annotations. Use a new output directory.

### 1. Retargeting and Rendering

[render_dexycb.py](dex-retargeting-main/example/position_retargeting/render_dexycb.py) retargets the annotated fingertips to the Ability Hand and renders the robot with the scene objects using the source camera intrinsics.

```bash
conda activate handedit-retargeting
python dex-retargeting-main/example/position_retargeting/render_dexycb.py \
  --view data/dex-ycb/SUBJECT/CAPTURE/CAMERA \
  --frames 41 80 \
  --output work/clip
```

It writes source frames to `work/clip/rgb/` and robot renders to `work/clip/rendered/`, keeping the source frame IDs.

### 2. Background Preparation

[segment.py](sam3-main/segment.py) tracks the human hand and exports binary masks. [batch_inference.py](ProPainter-main/batch_inference.py) calls [inference.py](ProPainter-main/inference.py) to remove the masked hand with ProPainter.

```bash
conda activate handedit-synthesis
python sam3-main/segment.py human --input work/clip/rgb --output work/human_masks
python ProPainter-main/batch_inference.py \
  --photo_root work/clip/rgb \
  --mask_root work/human_masks \
  --output_root work/background
```

### 3. Robotic Hand Extraction

Run `segment.py robot` on the rendered frames to export RGBA robot cutouts.

```bash
python sam3-main/segment.py robot --input work/clip/rendered --output work/robot_cutouts
```

For interactive mask selection and correction, use [humanhand_sam.py](sam3-main/humanhand_sam.py) for the source hand or [sim_sam.py](sam3-main/sim_sam.py) for the rendered robot:

```bash
python sam3-main/humanhand_sam.py
python sam3-main/sim_sam.py
```

### 4. Composition and Blending

[composite.py](composite.py) erodes and feathers the robot alpha mask, smooths foreground RGB and blends the cutout with the inpainted background.

```bash
python composite.py \
  --fg_dir work/robot_cutouts \
  --bg_dir work/background \
  --output_dir work/composite \
  --fg_blur_strength 3 \
  --erosion_strength 3 \
  --blur_strength 12
```

- `fg_blur_strength`: foreground RGB smoothing; `3` uses a 5 x 5 Gaussian kernel.
- `erosion_strength`: alpha erosion to remove boundary artifacts; `3` applies a 3 x 3 kernel three times.
- `blur_strength`: alpha feathering for blending; `12` uses a 13 x 13 Gaussian kernel.

### 5. Image Harmonization

[harmonize.py](harmonize.py) loads the HandEdit Harmonizer checkpoint and adjusts temperature, brightness, contrast, saturation, highlights and shadows inside the robot mask. Pixels outside the mask keep their composite values.

```bash
python harmonize.py \
  --input work/composite \
  --masks work/robot_cutouts \
  --output work/clip/edited
```

The paired training images are `work/clip/rgb/000041.jpg` and `work/clip/edited/000041.png`, and so on.

## Acknowledgments

We thank [SAM 3](https://github.com/facebookresearch/sam3), [ProPainter](https://github.com/sczhou/ProPainter), [dex-retargeting](https://github.com/dexsuite/dex-retargeting), [dex-urdf](https://github.com/dexsuite/dex-urdf), [SAPIEN](https://github.com/haosulab/SAPIEN), [Harmonizer](https://github.com/ZHKKKe/Harmonizer), [DexYCB](https://github.com/NVlabs/dex-ycb-toolkit), [manopth](https://github.com/hassony2/manopth) and [MANO](https://mano.is.tue.mpg.de/) for their code, models and data.

## License

Our original code is licensed under the [MIT License](LICENSE).
Third-party components retain their respective licenses.
