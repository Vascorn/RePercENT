# RePercENT: Scaling Disentangled Representation Learning Beyond Two Modalities 

[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)](https://arxiv.org/search/?query=RePercENT%20Scaling%20Disentangled%20Representation%20Learning%20Beyond%20Two%20Modalities&searchtype=all)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-core-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of __RePercENT__, a multimodal representation learning framework for disentangling modality-specific and shared representations across more than two modalities.

RePercENT takes pre-extracted Foundation Model (__FM__) embeddings as input. For each modality pair $ (i, j) $, it learns modality-specific components $ \mathbf{u}_{ij} $ and $ \mathbf{u}_{ji} $, together with shared components $ \mathbf{s}_{ij} $ and $ \mathbf{s}_{ji} $. The framework is agnostic to both the input modalities and the backbone FMs, allowing it to operate on embeddings from arbitrary modality sets and encoders.

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

The repository includes a Dockerfile and `compose.yaml` configuration file, which provides an easy minimal `docker build` and `docker run` commands.

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

# 📁 Data layout

Dataset files are excluded from the repository. For the synthetic dataset one can easily regenerate on-the-fly the different synthetic datasets, using the same configuration setup provided in the [`configs/data/synthetic_data_*m.yaml`](configs/data/), while for the two datasets we provide the dedicated preprocessing pipelines used in the [`src/utils/`](src/utils/)

See [`data/README.md`](data/README.md) for the expected local directory layout.

# 🏃 Quick Start

1. Start with the lightweight demo in [`training/demo.py`](training/demo.py). It generates random tensors for each modality, mimicking pre-extracted embeddings $Z_i \in \mathbb{R}^{S_i \times E_i}$, where $S_i$ is the sequence length and $E_i$ is the embedding dimension. The script then trains RePercENT with the same end-to-end objective used by the full pipelines, making it the shortest example of how to plug arbitrary modality embeddings into the model.

Inside an interactive container, run:

```bash
python training/demo.py --M 2 --base_seed 2
```

Demo configs are provided for `--M 2` and `--M 3`, where `M` is the number of modalities.

2. For synthetic experiments, no external preprocessing is required. The training script can generate data from the provided [`configs/`](configs/) or load a saved synthetic dataset. To generate data on the fly and train RePercENT:

```bash
python training/main.py \
  --no-load_data \
  --model_type repercent \
  --k1 3 \
  --k2 2 \
  --base_seed 2
```

Use `--model_type jointopt` for the JointOpt baselines.

3. For the real-world IRFL and HONeYBEE/TCGA experiments, generate or place the expected train/test tensors before launching training. See [`data/README.md`](data/README.md) for data preparation and [`training/README.md`](training/README.md) for the training entry points.

# 🔐 Available upon request

For exact reproducibility fo the results, we can provide upon request:

- The exact preprocessed HONeYBEE train/ test splits used.
- The final train-test tensor and augmented views for the IRFL detection pipeline.
- The used generated synthetic datasets. 

# 📝 Citation
_To be anounced_

# 🤝 Acknowledgements

Parts of this repository adapt code from: <a href="https://github.com/uhlerlab/DisentangledSSL/tree/master"><img src="https://cdn.simpleicons.org/github/58a6ff" width="14" alt="GitHub"> DisentangledSSL</a>, and for the gMLP baseline we use third-party code <a href="https://github.com/jaketae/g-mlp"><img src="https://cdn.simpleicons.org/github/58a6ff" width="14" alt="GitHub"> g-mlp</a> under src/models/third_party/ which provides a PyTorch implementation for [Pay Attention to MLPs](https://arxiv.org/abs/2105.08050).
