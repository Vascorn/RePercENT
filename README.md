# RePercENT: Scaling Disentangled Representation Learning Beyond Two Modalities 

[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)](https://arxiv.org/search/?query=RePercENT%20Scaling%20Disentangled%20Representation%20Learning%20Beyond%20Two%20Modalities&searchtype=all)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-core-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of **RePercENT**, a multimodal disentangled representation learning framework designed to scale beyond the two-modality setting. The repository includes model implementations, synthetic and real-world training pipelines, JointOpt baselines, and posthoc evaluation scripts.

![Model overview](assets/image.png)

## Repository Map

| Path | Contents |
| --- | --- |
| [`src/`](src/README.md) | Core RePercENT, Perceiver, JointOpt, data utilities, and adapted third-party components. |
| [`training/`](training/README.md) | Training entry points for synthetic, IRFL, and TCGA/HONeYBEE experiments. |
| [`configs/`](configs/) | Model, data, training, and posthoc analysis configuration files. |
| [`posthoc/synthetic/`](posthoc/synthetic/README.md) | Synthetic experiment evaluation, probes, complexity, and summary plots. |
| [`posthoc/irfl/`](posthoc/irfl/README.md) | IRFL detection-task evaluation and embedding visualizations. |
| [`posthoc/honeybee/`](posthoc/honeybee/README.md) | TCGA/HONeYBEE cancer-type probes, baselines, visualizations, and missing modality analysis. |
| [`fine_tuning/`](fine_tuning/) | CLIP fine-tuning helpers for IRFL-related experiments. |

## Setup

```bash
pip install -r requirements.txt
```

## Upon request

- The preprocessed HONeYBEE train/ test split.
- The final train-test tensor and augmented views for the IRFL detection pipeline.
- The used generated synthetic datasets. 

## Citation
Preprint coming soon.

```bibtex
@misc{repercent2026,
  title = {RePercENT: Scaling Disentangled Representation Learning Beyond Two Modalities},
  author = {TBD},
  year = {2026},
  note = {Preprint coming soon}
}
```

## Acknowledgements

Parts of this repository adapt code from: <a href="https://github.com/uhlerlab/DisentangledSSL/tree/master"><img src="https://cdn.simpleicons.org/github/58a6ff" width="14" alt="GitHub"> DisentangledSSL</a>, and the gMLP baseline uses vendored third-party code under src/models/third_party/.