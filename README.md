# Hierarchical-Pose-Prompting-for-Generalized-Animal-Pose-Estimation
Official code release for Hierarchical Pose Prompting for Generalized Animal Pose Estimation (MM 2026), built on the MMPose 1.3.2 codebase and API layout.

The current release is organized around the AP-10K configuration:

```text
configs/animal_2d_keypoint/topdown_heatmap/ap10k/hpp.py
```

The research `configs/` directory is retained as provided. For the current release, `hpp.py` should be treated as the reference configuration for the HPP implementation.

## Method-to-code map

The main HPP components are implemented in the following files:

- **HPP estimator / training losses:** `mmpose/models/pose_estimators/hpp3.py`
- **Pose Region Activation (PRA):** `mmpose/models/pose_estimators/pra3.py`
- **Spatial-aware Scaled Cross-modal Fusion (SSCF):** `mmpose/models/pose_estimators/kpa3.py`
- **CLIP vision/text encoders and adaptive keypoint prompt:** `mmpose/models/backbones/clip.py`
- **Prompt loading from annotation JSON:** `mmpose/datasets/datasets/base/base_coco_style_dataset.py`
- **Heatmap loss:** `mmpose/models/losses/heatmap_loss.py`

The registry name remains `hpp3`, which is the model type used by `hpp.py`.

## Installation

Please prepare the PyTorch/OpenMMLab environment by following the **latest official MMPose installation guide**:

- [MMPose Installation](https://mmpose.readthedocs.io/en/latest/installation.html)

After the MMPose dependencies are available, install this repository in editable mode:

```bash
pip install -v -e .
```

The HPP code also uses OpenAI CLIP, `timm`, and the augmentation dependencies referenced by the configuration. Install them if they are not already available in your environment.

## MMPose APIs and demos

The upstream-style MMPose API and the demos relevant to this top-down animal-pose codebase are retained in the release:

```text
mmpose/apis/
demo/image_demo.py
demo/inferencer_demo.py
demo/topdown_demo_with_mmdet.py
demo/mmdetection_cfg/rtmdet_m_8xb32-300e_coco.py
```

`model-index.yml` and `dataset-index.yml` are also retained so the standard MMPose/MIM resource layout remains available when packaging the repository.

For HPP itself, the reference training/testing path is still `hpp.py` together with the prompt-augmented annotation JSONs described below. The generic upstream MMPose demos do not generate HPP pose prompts automatically; when using the HPP estimator, each sample must provide the `instance_text_prompt` metadata expected by the model.

## Datasets and pose prompts

The experiments use **AP-10K** and **Animal-Kingdom**. Please prepare the original datasets according to their official releases / the corresponding MMPose dataset format.

Prompt-augmented annotation JSON files for **AP-10K** and **Animal-Kingdom** are provided in:

- [pose prompt.zip](https://github.com/abandonxv/Hierarchical-Pose-Prompting-for-Generalized-Animal-Pose-Estimation/blob/main/pose%20prompt.zip)

Download and extract the archive, then place the JSON files under the annotation directories referenced by the corresponding configs.

For the reference AP-10K config `hpp.py`, the expected files are:

```text
data/ap10k/
├── data/
└── annotations/
    ├── blip2train.json
    └── blip2val.json
```

Each prompt-augmented image record contains a `prompt` field. The dataset loader passes this text to the model as `instance_text_prompt`.

The paths inside the research configs may point to the authors' local filesystem (for example `/root/mmpose/data/...`). Adjust dataset and pretrained-model paths locally when running the code.

## Pretrained CLIP weights

The reference config uses a local CLIP ViT-B/16 checkpoint path. Prepare the corresponding CLIP checkpoint and update or mirror the path used by your local copy of the config as needed.

## Training

Single process:

```bash
python tools/train.py configs/animal_2d_keypoint/topdown_heatmap/ap10k/hpp.py
```

Distributed training, for example with 2 GPUs:

```bash
bash tools/dist_train.sh configs/animal_2d_keypoint/topdown_heatmap/ap10k/hpp.py 2
```

## Testing

```bash
python tools/test.py \
  configs/animal_2d_keypoint/topdown_heatmap/ap10k/hpp.py \
  /path/to/checkpoint.pth
```

## Repository scope

This release keeps the HPP implementation together with the MMPose API/demo/resource files needed for normal source-tree use. Confirmed non-upstream experimental modules that are unrelated to the current `hpp.py` execution path have been removed. The `configs/` directory is intentionally left byte-for-byte unchanged in this cleanup pass; therefore, historical research configs that reference removed experimental modules are retained as records but are not part of the supported release path. The supported reference path is `hpp.py`.

## Acknowledgement

This repository is built upon [MMPose](https://github.com/open-mmlab/mmpose). We sincerely thank the MMPose team for providing the open-source pose estimation framework and infrastructure used in this project.

We also thank the authors of the following works for releasing their code and for their contributions to language-guided animal pose estimation:

- **CLAMP (CVPR 2023)** — [GitHub](https://github.com/xuzhang1199/CLAMP)
- **PPAP (CVPR 2025)** — [GitHub](https://github.com/Raojiyong/PPAP)
## Citation

If this work is useful for your research, please cite the paper. Citation metadata is provided in `CITATION.cff`.
