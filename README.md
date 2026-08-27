# GorillaWatch: An Automated System for In-the-Wild Gorilla Re-Identification and Population Monitoring

**Paper accepted at WACV 2026** | [arXiv:2512.07776](https://arxiv.org/abs/2512.07776)

## Overview

GorillaWatch is a comprehensive automated system for in-the-wild gorilla re-identification and population monitoring. Monitoring critically endangered western lowland gorillas has historically required immense manual effort to re-identify individuals from vast archives of camera trap footage. This work addresses this challenge by introducing an end-to-end pipeline integrating detection, tracking, and re-identification. We leverage multi-frame self-supervised pretraining and demonstrate that large-scale image backbones outperform specialized video architectures for this task.

## Key Contributions

- **Novel Benchmark Datasets**: Three large-scale, in-the-wild datasets for gorilla analysis:
  - [**Gorilla-SPAC-Wild**](https://huggingface.co/datasets/gorilla-watch/Gorilla-SPAC-Wild): Largest video dataset for wild primate re-identification
  - [**Gorilla-Zoo-Berlin**](https://huggingface.co/datasets/gorilla-watch/Gorilla-Zoo-Berlin): Cross-domain re-identification generalization assessment
  - [**Gorilla-SPAC-MoT**](https://huggingface.co/datasets/gorilla-watch/Gorilla-SPAC_MoT): Multi-object tracking in camera trap footage

- **End-to-End Detection & Tracking Pipeline**: Integrated framework for automatic gorilla detection, tracking, and re-identification from video

- **Multi-Frame Self-Supervised Pre-training**: Leverages temporal consistency in tracklets to learn domain-specific features without manual labels

- **Interpretability Verification**: Differentiable adaptation of AttnLRP ensures the model relies on discriminative biometric traits rather than background correlations

- **Large-Scale Backbone Analysis**: Demonstrates that aggregating features from large-scale image backbones outperforms specialized video architectures

- **Unsupervised Population Counting**: Spatiotemporal constraints integrated into clustering to mitigate over-segmentation for accurate population monitoring

## Architecture

The GorillaWatch system comprises:

- **Detection & Tracking**: Automatic gorilla detection and temporal tracking from video streams
- **Re-Identification Backbone**: Large-scale image backbones (Vision Transformer, ConvNets) for learning discriminative person/gorilla embeddings
- **Multi-Frame Self-Supervised Learning**: Temporal pretraining that leverages consistency across frames in tracklets
- **Population Monitoring**: Constrained clustering with spatiotemporal constraints for accurate gorilla counting and re-identification
- **Interpretability**: AttnLRP-based verification to ensure predictions rely on valid biometric features (distinctive markings, body shape) rather than spurious background correlations

## Installation

### Requirements

- Docker (with the NVIDIA container runtime for GPU support)

### Setup

All commands run inside the container built from the provided `Dockerfile`. The image is built automatically the first time you use the helper script:

```bash
# Open a shell in the container on GPU 0
./scripts/run-in-docker.sh -g 0

# Or run a script directly, e.g. training on GPU 1
./scripts/run-in-docker.sh -g 1 ./scripts/train.sh
```

The project is mounted at `/workspaces/gorillawatch`, `pip install -e .` runs before your command. W&B credentials are picked up from `$WANDB_API_KEY` or `~/.netrc` (run `wandb login` once on the host).

Building the image requires [BuildKit](https://docs.docker.com/build/buildkit/)/`buildx` to be installed.

## Project Structure

```
src/gorillawatch/
├── data/                    # Data loading and dataset utilities
├── model/                   # Model architectures and training logic
├── clustering/              # Constrained clustering for evaluation
├── losses/                  # Triplet loss and regularization implementations
├── qualitative_evaluate/    # Qualitative analysis utilities
├── utils/                   # Determinism and type helpers
├── train_and_eval.py       # Main training entry point
└── evaluate.py             # Evaluation and inference
```

## Quick Start

### Training a Model

The easiest way to train a model is using the provided shell script:

```bash
./scripts/train.sh
```

This trains a ViT-Small DINOv2 model on the Gorilla-SPAC-Wild dataset with paper-compliant hyperparameters:
- **Batch size**: 8 (effective: 48 with 6 gradient accumulation steps)
- **Epochs**: 100 with early stopping (patience=10)
- **Learning rate**: 1.9×10⁻⁷ (cosine schedule)
- **Regularization**: L2=0.0059, L2SP=1.3×10⁻⁵
- **Loss**: Hard triplet mining with margin=0.647
- **Evaluation frequency**: Every 10 epochs (configurable via `EVAL_FREQUENCY`)

You can customize parameters by editing variables in `scripts/train.sh` or run the Python script directly:

```bash
python src/gorillawatch/train_and_eval.py \
  --wandb_project "GorillaWatch-Training" \
  --wandb_run "my_experiment" \
  --backbone_name vit_small_patch14_dinov2.lvd142m \
  --dataset "gorilla-watch/Gorilla-SPAC-Wild" \
  --dataset_config "face_with_body" \
  --image_size 518
  --epochs 100 \
  --batch_size 8 \
  --lr 0.00000019 \
  --eval_frequency 10
```

The dataset will be automatically downloaded from HuggingFace on first run.

### Evaluating Fine-Tuned Models from HuggingFace

To evaluate our models published on huggingface use the prepared bash script:

```bash
./scripts/eval_hf_fine_tuned.sh
```

Or directly via the python script:

```bash
python src/gorillawatch/evaluate.py \
  --hf_model "gorilla-watch/GorillaWatch-DINOv2-Giant" \
  --dataset "gorilla-watch/Gorilla-SPAC-Wild" \
  --dataset_config "face_with_body" \
  --batch_size 8
```

### Evaluating a Fine-Tuned Model

Evaluate a local trained checkpoint:

```bash
./scripts/eval_local_fine_tuned.sh
```

Or evaluate specific checkpoints:

```bash
python src/gorillawatch/evaluate.py \
  --evaluate_model_path saved_checkpoints/your_model.pth \
  --backbone_name vit_small_patch14_dinov2.lvd142m \
  --dataset "gorilla-watch/Gorilla-SPAC-Wild" \
  --dataset_config "face" \
  --batch_size 8
  --image_size 518
```

### Zero-Shot Evaluation

Evaluate pre-trained backbones without fine-tuning:

```bash
./scripts/eval_zero_shot.sh
```

### Clustering Analysis

```bash
./scripts/clustering.sh
```


## Experimental Results

The paper demonstrates:
- State-of-the-art re-identification performance on Gorilla-SPAC-Wild
- Cross-domain generalization assessment using Gorilla-Berlin-Zoo
- Multi-object tracking evaluation on Gorilla-SPAC-MoT
- Interpretability analysis via AttnLRP showing reliance on biometric features
- Comparative analysis of backbone architectures for gorilla re-identification
- Population counting accuracy using spatiotemporal-constrained clustering

Detailed results and ablation studies are available in the [full paper](https://arxiv.org/abs/2512.07776).


## Citation

If you use GorillaWatch in your research, please cite our WACV 2026 paper:

```bibtex
@inproceedings{GorillaWatch2026,
  title={GorillaWatch: An Automated System for In-the-Wild Gorilla Re-Identification and Population Monitoring}, 
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  author={Maximilian Schall and Felix Leonard Knöfel and Noah Elias König and Jan Jonas Kubeler and Maximilian von Klinski and Joan Wilhelm Linnemann and Xiaoshi Liu and Iven Jelle Schlegelmilch and Ole Woyciniuk and Alexandra Schild and Dante Wasmuht and Magdalena Bermejo Espinet and German Illera Basas and Gerard de Melo},
  year={2026},
  archivePrefix={arXiv},
  eprint={2512.07776}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Acknowledgments

The project on which this report is based was funded by the Federal Ministry of Research, Technology and Space under the funding code “KI-Servicezentrum Berlin-Brandenburg” 16IS22092. We acknowledge the support of Sabine Plattner African Charities (SPAC) for their funding to this research. We are grateful to Zoo Berlin for their expert assistance and facility access. This collaboration enabled the development of AI tools capable of being deployed in the wild to directly support gorilla conservation. The responsibility for the content of this publication remains with the authors.