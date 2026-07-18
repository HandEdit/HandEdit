<div align="center">

<div style="display: flex; align-items: center; justify-content: center; gap: 20px;">
  <h1 style="margin: 0;">HandEdit: A Unified Benchmark for Egocentric Human-to-Robot Dexterous Hand Image Editing</h1>
</div>
</div>



## Overview
We present **HandEdit**, a large-scale embodiment-aware image-editing dataset and benchmark for transforming human hands and arms into dexterous robotic embodiments in egocentric scenes. HandEdit contains over **200M editing instances** from five source datasets, covering **26 URDF embodiments**: 13 hand-only and 13 hand-arm configurations. We further define two URDF-conditioned benchmark tracks, **Hand-only** and **Hand-Arm**, and evaluate 11 representative image-editing baselines with generic similarity metrics, VLM-based judgment, and embodiment-aware metrics. HandEdit bridges image editing and robotics by supporting embodiment-aware editing and scalable robot-centric data generation for dexterous embodied learning.

![overview](./assets/handedit_teaser.png)


## Comparison on the Hand-only and Hand-Arm Tracks
We visualize normalized scores across generic similarity metrics, embodiment-aware metrics, and VLM-based judgment. Metrics marked with ∗ are originally lower-is-better metrics; their scores are inverted for visualization so that larger values consistently indicate better performance.

![Comparison](./assets/radar.png)

##  HandEdit Dataset & Benchmark

### Dataset
Sample data are available for download on <img src="./assets/hf-logo.png" alt="Hugging Face" width="18"/> [HandEdit Dataset](https://huggingface.co/datasets/HandEdit/HandEdit).


### Two benchmark tracks:
- `hand-only`: replace the human hand with a target robot hand.
- `hand-Arm`: replace the visible hand-arm region with a target robot arm-hand embodiment.

The track is set by `replacement_scope` in the manifest.


## HandEdit Evaluation Tools

### Install

```bash
conda create -n handedit-eval python=3.10 -y
conda activate handedit-eval
pip install -r requirements.txt
```

DINOv2 and CLIP checkpoints are not included. Download them separately and pass the local paths to `eval.py`.

### Manifest

The evaluator reads a JSONL manifest, one sample per line. A typical record is:

```json
{"id":"000001","replacement_scope":"hand-only","target_name":"Shadow Hand","src_path":"data/src/000001.png","pred_path":"data/pred/000001.png","gt_path":"data/gt/000001.png","gt_mask_path":"data/gt_mask/000001.png","test_mask_path":"data/pred_mask/000001.png","human_mask_path":"data/human_mask/000001.png","robot_mask_path":"data/robot_mask/000001.png","object_mask_path":"data/object_mask/000001.png","urdf_ref_paths":["data/urdf/shadow/view_0.png","data/urdf/shadow/view_1.png"],"urdf_mask_paths":["data/urdf_mask/shadow/view_0.png","data/urdf_mask/shadow/view_1.png"]}
```

`urdf_ref_paths` stores the rendered URDF views for the target embodiment. `urdf_mask_paths` is optional; if it is missing, the full render is used. ROI metrics use `human_mask_path ∪ robot_mask_path`. If neither mask is available, the evaluator falls back to the full image.

### Build a manifest

```bash
python build_manifest.py \
  --src-root data/src \
  --pred-root data/pred \
  --gt-root data/gt \
  --gt-mask-root data/gt_mask \
  --test-mask-root data/pred_mask \
  --human-mask-root data/human_mask \
  --robot-mask-root data/robot_mask \
  --object-mask-root data/object_mask \
  --replacement-scope hand-only \
  --target-name "Shadow Hand" \
  --urdf-refs data/urdf/shadow/view_0.png data/urdf/shadow/view_1.png data/urdf/shadow/view_2.png data/urdf/shadow/view_3.png data/urdf/shadow/view_4.png data/urdf/shadow/view_5.png \
  --urdf-masks data/urdf_mask/shadow/view_0.png data/urdf_mask/shadow/view_1.png data/urdf_mask/shadow/view_2.png data/urdf_mask/shadow/view_3.png data/urdf_mask/shadow/view_4.png data/urdf_mask/shadow/view_5.png \
  --out-manifest manifests/shadow_hand_only.jsonl
```

For `hand-arm`, set `--replacement-scope hand-arm` and pass the matching arm-hand URDF renders.

### Run evaluation

```bash
python eval.py \
  --manifest manifests/shadow_hand_only.jsonl \
  --experiment shadow_hand_only \
  --output-dir runs \
  --device cuda \
  --shape-model models/dinov2 \
  --clip-model models/clip
```

Outputs are written to `runs/<experiment>/metrics/`.

### Metrics

The evaluator reports PSNR/SSIM/LPIPS on three part(Full-image, ROI, and Background), FID, Removal, Struct Fidelity, ID Fidelity, Interaction, and VLM scores.

`Struct Fidelity,` uses DINOv2 on the edited ROI and the pseudo-GT ROI. `ID Fidelity` combines two terms: max CLIP similarity between the edited ROI and the target URDF render bank, and masked pixel-wise CIE Lab similarity between the edited ROI and the pseudo-GT. The default weights are `0.5 / 0.5`, and the Lab temperature is `25`.

`Interaction` is computed on the object/contact region. If `object_mask_path` is missing, the evaluator uses a local band around the replacement ROI.