# RePercENT Source

[![Upstream](https://img.shields.io/badge/derived-DisentangledSSL-orange)](https://github.com/uhlerlab/DisentangledSSL/tree/master)
[![TCGA Dataset](https://img.shields.io/badge/Hugging%20Face-TCGA-FFD21E?logo=huggingface&logoColor=yellow)](https://huggingface.co/datasets/Lab-Rasool/TCGA)
[![IRFL Dataset](https://img.shields.io/badge/Hugging%20Face-IRFL-FFD21E?logo=huggingface&logoColor=yellow)](https://huggingface.co/datasets/lampent/IRFL)

This directory contains the core model implementations, data utilities, and adapted disentanglement baselines used by RePercENT. The training scripts and model usage lives in `../training`, while model and data configuration files live in `../configs`.

## Directory Map

| Path | Contents |
| --- | --- |
| `models/` | RePercENT, Perceiver encoders, JointOpt baselines, pretrained encoder helpers, and vendored gMLP code. |
| `utils/` | Synthetic data generation, IRFL/Honeybee preprocessing utilities, augmentations, metrics helpers, and probing utilities. |
| `DisentangledSSL/` | Adapted code derived from the upstream DisentangledSSL repository. |

## Model Components

| File | What it provides | Notes |
| --- | --- | --- |
| `models/repercent.py` | `RePercENT`, `DisenEncoder`, and `DisenLoss`. | Main multimodal disentanglement model. It uses one Perceiver-based encoder per modality and supports the generalized multi-modal objective. |
| `models/perceiver.py` | `Perceiver` and `PerceiverDisen`. | `Perceiver` is the default encoder backbone, that uses group slot attention and alternates between cross- and self- attention blocks. `PerceiverDisen` is an experimental variant that removes latent self-attention and encourages per-component specialization through Mixture of Experts FeedForward networks.<br><br>**Note:** All the results and experiments use the `Perceiver` adapted variant, however we also provide `PerceiverDisen` as it might be useful for further research purposes.|
| `models/jointopt.py` | `JointOpt`, `MLP`, and `GRUEncoder`. | Baseline family with separate encoders for unique and shared components. gMLP support is wired through `models/third_party/g_mlp_repo`. |
| `models/pretrained_encoders/` | CLIP embedding helpers. | Utilities for extracting or wrapping pretrained image/text embeddings, used for the IRFL detection task.|
| `models/third_party/g_mlp_repo/` | Vendored gMLP implementation. | Based on the upstream gated MLP implementation used by the JointOpt gMLP baseline. |

## Data And Utilities

| File | Purpose |
| --- | --- |
| `utils/synthetic_dataset.py` | Synthetic multimodal dataset generation. The main experiments use `GenerateTokenizedData`; `GenerateData` and `GeneratePermData` are kept as useful variants. |
| `utils/irfl_preprocess.py` | A preprocessing script containing a step-by-step complete pipeline for constructing the irfl dataset containing __images__, __captions__ and __definitions__. The script also pre-extracts random augmentations for image/ text view so that the final dataset does not require any forward pass through the backbone CLIP model.|
| `utils/irfl_dataset.py` | Dataset wrapper and construction helpers for IRFL tensors. |
| `utils/irfl_augmentations.py` | IRFL augmentation utilities. |
| `utils/honeybee_dataset.py` | TCGA cohort embedding extraction and preprocessing helpers for __molecular__, __clinical__, __pathology-report__, and __whole-slide-image__ modalities. |
| `utils/helpers.py` | Reproducibility helpers, latent extraction, linear probes, and plotting utilities for the synthetic experiments. |


## Configuration Pointers

Model architecture choices are controlled by YAML files in `../configs/model`:

- `repercent_*.yaml` selects Perceiver options such as depth, latent count, slot attention, and positional encoding.
- `jointopt_*.yaml`, `gru_*.yaml`, and `gmlp_*.yaml` configure the JointOpt baseline models.
- Data shape assumptions come from the matching files in `../configs/data`.

## Third-party and adapted code

The `DisentangledSSL/` folder contains derived or adapted code from the upstream [uhlerlab/DisentangledSSL](https://github.com/uhlerlab/DisentangledSSL/tree/master) project. Please consult the upstream repository for definitive licensing, citation, and usage details. Third-party code under `models/third_party/` keeps its own license and attribution files where available.
