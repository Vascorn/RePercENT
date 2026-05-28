# RePercENT: Scaling Disentangled Representation Learning Beyond Two Modalities 

[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)](https://arxiv.org/search/?query=RePercENT%20Scaling%20Disentangled%20Representation%20Learning%20Beyond%20Two%20Modalities&searchtype=all)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-core-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of __RePercENT__, a multimodal representation learning framework for disentangling modality-specific and shared representations across more than two modalities.

RePercENT takes pre-extracted Foundation Model (__FM__) embeddings as input. For each modality pair $ (i, j) $, it learns modality-specific components $\mathbf{u}_{ij}$ and $\mathbf{u}_{ji}$, together with shared components $\mathbf{s}_{ij}$ and $\mathbf{s}_{ji}$. The framework is agnostic to both the input modalities and the backbone FMs, allowing it to operate on embeddings from arbitrary modality sets and encoders.

In addition to the RePercENT model, this repository provides implementations of JointOpt baseline alternatives, which use the same training regime but separate encoders for each representation component. We include three JointOpt variants: MLP, GRU, and gMLP. 

The codebase also provides the synthetic data generation and experiment pipeline implementation, as well as the real-world dataset preparation and posthoc evaluation scripts.

![Model overview](.github/image.png)


# ⚙️ Setup

Clone the repo, including the gMLP submodule:

```bash
git clone --recurse-submodules https://github.com/Vascorn/RePercENT.git
cd RePercENT
```

If the repo is already cloned

```bash
git submodule update --init --recursive
```

The repository includes a Dockerfile and `compose.yaml` configuration file, which wraps the usual `docker build` and `docker run` commands.

To build the image:

```bash
docker compose build repercent
```

To start an interactive container:

```bash
docker compose run --rm repercent
```

On a Linux machine with NVIDIA GPU, use the following:

```bash
docker compose run --rm repercent-gpu
```

By default, `compose` tags the image as `repercent`. To use a different image name, set `REPERCENT_IMAGE`:

```bash
REPERCENT_IMAGE=<repercent_image_name> docker compose build repercent
REPERCENT_IMAGE=<repercent_image_name> docker compose run --rm repercent
REPERCENT_IMAGE=<repercent_image_name> docker compose run --rm repercent-gpu
```

> [!IMPORTANT]
> On Apple Silicon Macs, the default `linux/amd64` platform runs under emulation and is intended for smoke tests, and small CPU runs. Full CUDA training should be run on a Linux machine with NVIDIA Docker support. GPU access is kept in the explicit `repercent-gpu` service so the default service remains portable across Mac, CPU-only Linux, and GPU Linux hosts.

> [!NOTE]
> For running, `WANDB_MODE=offline` by default so local tests do not upload runs to Weights & Biases. To change this, set `WANDB_MODE` to `online` and add your `WANDB_API_KEY`, for example:
>
> ```bash
> WANDB_MODE=online WANDB_API_KEY=<your-key> REPERCENT_IMAGE=<repercent_image_name> docker compose run --rm repercent
> ```

<!-- 

## Expected Data Layout

Dataset files are not tracked in git and are excluded from Docker image builds.
See [`data/README.md`](data/README.md) for the expected local directory layout. -->

# 🗺️ Repository Map

| Path | Contents |
| --- | --- |
| [`src/`](src/README.md) | Core RePercENT, Perceiver, JointOpt, data utilities, and adapted third-party components. |
| [`training/`](training/README.md) | Training entry points for synthetic, IRFL, and TCGA/HONeYBEE experiments. |
| [`configs/`](configs/) | Model, data, training, and posthoc analysis configuration files. |
| [`posthoc/synthetic/`](posthoc/synthetic/README.md) | Synthetic experiment evaluation, probes, complexity, and summary plots. |
| [`posthoc/irfl/`](posthoc/irfl/README.md) | IRFL detection-task evaluation and embedding visualizations. |
| [`posthoc/honeybee/`](posthoc/honeybee/README.md) | TCGA/HONeYBEE cancer-type probes, baselines, visualizations, and missing modality analysis. |
| [`fine_tuning/`](fine_tuning/) | CLIP fine-tuning helpers for IRFL-related experiments. |


# 🏃 Quick Start

# 🔐 Available upon request

- The preprocessed HONeYBEE train/ test split.
- The final train-test tensor and augmented views for the IRFL detection pipeline.
- The used generated synthetic datasets. 

# 📝 Citation
_To be anounced_

# 🤝 Acknowledgements

Parts of this repository adapt code from: <a href="https://github.com/uhlerlab/DisentangledSSL/tree/master"><img src="https://cdn.simpleicons.org/github/58a6ff" width="14" alt="GitHub"> DisentangledSSL</a>, and for the gMLP baseline we use third-party code <a href="https://github.com/jaketae/g-mlp"><img src="https://cdn.simpleicons.org/github/58a6ff" width="14" alt="GitHub"> g-mlp</a> under src/models/third_party/ which provides a PyTorch implementation for [Pay Attention to MLPs](https://arxiv.org/abs/2105.08050).
