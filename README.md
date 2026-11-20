# HandEdit Preprocessing

[Project](https://handedit.github.io/) | [Dataset](https://huggingface.co/datasets/HandEdit/HandEdit) | [Models](https://huggingface.co/HandEdit/HandEdit-LoRA)

Code for the HandEdit data processing pipeline. We retarget human hand motion to a robot hand, remove the human hand from the source images, and composite and harmonize the rendered robot with the scene.

This branch includes our modified **SAM 3**, **ProPainter** and **dex-retargeting**. We use one [DexYCB](https://dex-ycb.github.io/) camera clip with a right Ability Hand as an example.

```text
.
|-- dex-retargeting-main/  # Hand motion retargeting and rendering
|-- sam3-main/             # Human hand masks and robot cutouts
|-- ProPainter-main/       # Background inpainting
|-- composite.py           # Composition and blending
`-- harmonize.py           # Image harmonization
```

## Installation

Rendering and image editing use separate environments because dex-retargeting and SAM 3 require different NumPy versions.

```bash
# Rendering
conda create -n handedit-render python=3.10 -y
conda activate handedit-render
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -e ./dex-retargeting-main sapien==3.0.1 "setuptools<82"

# Image editing
conda create -n handedit-edit python=3.12 -y
conda activate handedit-edit
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt -r ProPainter-main/requirements.txt -e ./sam3-main "setuptools<82" kornia==0.8.2
```

Place the [SAM 3 checkpoint](https://huggingface.co/facebook/sam3) at `weights/sam3.pt` and our [Harmonizer checkpoint](https://github.com/HandEdit/HandEdit/blob/main/harmonizer/harmonizer_hand.pth) at `weights/harmonizer_hand.pth`. ProPainter downloads its pretrained weights on first use; they are also available from the [official release](https://github.com/sczhou/ProPainter/releases/tag/v0.1.0).

## Data Preparation

Download the sequences, calibration files and object models from [DexYCB](https://dex-ycb.github.io/), keeping its original layout:

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

The commands below illustrate one clip. Replace `SUBJECT`, `CAPTURE` and `CAMERA` with a sequence containing a visible right hand (`mano_sides: [right]` in `meta.yml`). Select at least two consecutive annotated frames and use a new output directory.

## Data Processing Pipeline

### 1. Retargeting and Rendering

Retarget the annotated fingertips to the Ability Hand. Render the robot and objects together with the source camera intrinsics to retain object occlusion.

```bash
conda activate handedit-render
python dex-retargeting-main/example/position_retargeting/render_dexycb.py \
  --view data/dex-ycb/SUBJECT/CAPTURE/CAMERA \
  --frames 41 80 \
  --output work/clip
```

The output contains matching `rgb/` and `rendered/` frames, named by their source frame IDs.

### 2. Background Preparation

Segment the human hand with SAM 3, then use ProPainter to remove it and reconstruct the background from neighboring frames.

```bash
conda activate handedit-edit
python sam3-main/segment.py human --input work/clip/rgb --output work/human_masks
python ProPainter-main/batch_inference.py \
  --photo_root work/clip/rgb \
  --mask_root work/human_masks \
  --output_root work/background
```

### 3. Robotic Hand Extraction

Segment the rendered robot hand and save RGBA cutouts with the visible robot region as alpha.

```bash
python sam3-main/segment.py robot --input work/clip/rendered --output work/robot_cutouts
```

The interactive tools [humanhand_sam.py](sam3-main/humanhand_sam.py) and [sim_sam.py](sam3-main/sim_sam.py) are included for mask selection and correction.

### 4. Composition and Blending

Erode and feather the robot alpha mask, smooth foreground RGB, and composite the robot over the inpainted background.

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

Apply the HandEdit Harmonizer checkpoint to adjust temperature, brightness, contrast, saturation, highlights and shadows inside the robot mask. Pixels outside the mask retain their composite values.

```bash
python harmonize.py \
  --input work/composite \
  --masks work/robot_cutouts \
  --output work/clip/edited
```

The resulting training pairs are the source frames in `work/clip/rgb/` and the corresponding edited frames in `work/clip/edited/`.

## Acknowledgments

We thank [SAM 3](https://github.com/facebookresearch/sam3), [ProPainter](https://github.com/sczhou/ProPainter), [dex-retargeting](https://github.com/dexsuite/dex-retargeting), [dex-urdf](https://github.com/dexsuite/dex-urdf), [SAPIEN](https://github.com/haosulab/SAPIEN), [Harmonizer](https://github.com/ZHKKKe/Harmonizer), [DexYCB](https://github.com/NVlabs/dex-ycb-toolkit), [manopth](https://github.com/hassony2/manopth) and [MANO](https://mano.is.tue.mpg.de/) for their code, models and data.

Third-party code retains its original licenses and copyright notices. See [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES) and the license files in each model directory.
