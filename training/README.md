# Training Pipelines

This directory contains the training pipelines for __RePercENT__ and the __JointOpt__ variants. For training hyperparameters, see `../configs/training`; for model architecture settings, see `../configs/model`.

## Directory Map

The directory's logic can be split into the following main components:

* `main_*.py` scripts are executable entry points for full training runs.
* `train_*.py` scripts provide model factories, training loops, test loops, and evaluation helpers used by the entry points.
* `log_data.py` provides Weights & Biases logging utilities for model details, checkpoints, and dataset artifacts.
* `demo.py` provides a generic/ minimal pipeline for training and testing RePercENT on mock pre-extracted embeddings.

### Entry points

| File | Description |
| --- | --- |
| `main.py` | Main synthetic-data training entry point for RePercENT and JointOpt baselines. The script currently uses `M = 2` internally and loads the matching synthetic data, model, and training configs. The script trains the model according to the loaded `../configs` recipe and evaluates the final model on the test set using linear probing. |
| `main_ablations_alpha.py` | Synthetic $ \alpha $-ablation entry point across selected modality counts. |
| `main_irfl.py` | IRFL training entry point for RePercENT, gMLP, and GRU variants. It includes model training and evaluation on the reserved test set on the IRFL detection task. |
| `main_honeybee.py` | Honeybee/TCGA training entry point for RePercENT, gMLP, and GRU variants. |


### Utilities

| File | Description |
| --- | --- |
| `train_repercent.py` | Shared RePercENT helpers across all datasets, including data splitting, dataloaders, and model construction. The script also provide the default generic train/test loops, and probe logging for the synthetic experiments. |
| `train_jointopt.py` | JointOpt model factory and encoder builder for MLP, GRU, and gMLP baselines. This is a generic script valid for all three types of experiments. |
| `train_irfl.py` | IRFL-specific train/test loops, retrieval metrics, distractor handling. |
| `train_honeybee.py` | Honeybee/TCGA-specific batch preparation, training loop, final evaluation, and cancer-type probing helpers. |

## How to use: 

### Synthetic Experiments

![Dataset: Synthetic](https://img.shields.io/badge/dataset-Synthetic-pink)
![Script](https://img.shields.io/badge/script-main.py-lightgrey)


To train on the synthetic dataset, use the main synthetic entry point:

```bash
python training/main.py \
  --load_data \
  --model_type repercent \
  --k1 3 \
  --k2 2 \
  --base_seed 2
``` 

Use `--no-load_data` to generate a new synthetic dataset instead of loading a saved one. The `--load_data`, `--save_data`, `--save_data_split`, and `--log_dataset_artifact` options additionally control the logging/ loading of the synthetic datasets

---

### IRFL experiments

![Dataset: IRFL](https://img.shields.io/badge/dataset-IRFL-blue)
![Script](https://img.shields.io/badge/script-main__irfl.py-lightgrey)

To train on IRFL, use the preprocessed IRFL tensors and select the model variant:

```bash
python training/main_irfl.py \
  --datasets_path ../data/irfl/datasets/ \
  --model_type repercent \
  --n_seeds 5 \
  --base_seed 2 \
  --comp_mod 1
```

Use `--model_type gmlp` or `--model_type gru` for the JointOpt baselines. `--comp_mod 1` evaluates image-caption retrieval; `--comp_mod 2` evaluates image-definition retrieval and is only meaningful when the 3-modality IRFL setup is used. Semantic encoding and group slot attention are enabled by default; use `--no-add_SE` or `--no-add_GSA` to disable them.

---

### HONeYBEE/ TCGA cohort

![Dataset: HONeYBEE](https://img.shields.io/badge/dataset-HONeYBEE-green)
![Script](https://img.shields.io/badge/script-main__honeybee.py-lightgrey)

To train on the Honeybee/TCGA cohort, use the Honeybee entry point and select the model variant:

```bash
python training/main_honeybee.py \
  --datasets_path ../data/honeybee/datasets/ \
  --model_type repercent \
  --n_seeds 5 \
  --base_seed 2 \
  --split_seed 42 \
  --wsi_embedding_mode slide \
  --filter_cancer_types TCGA-BRCA TCGA-COAD TCGA-GBM TCGA-HNSC TCGA-KIRC TCGA-LGG TCGA-LUAD TCGA-LUSC TCGA-OV TCGA-PRAD
```

The script loads a saved stratified split by default. Avoid passing string values such as `--load_test_split False`: with the current argparse definition, non-empty strings are parsed as `True`. Use the default saved split for reproducible runs, or update the script before using CLI booleans to create new splits.

---
### Alpha ablations

![Ablation](https://img.shields.io/badge/experiment-alpha_ablation-purple)
![Script](https://img.shields.io/badge/script-main__ablations__alpha.py-lightgrey)

To run the RePercENT alpha ablations across modality counts, provide the alpha values and modality counts explicitly:

```bash
python training/main_ablations_alpha.py \
  --model_type repercent \
  --alpha_values 0.01 0.1 1.0 10.0 100.0 \
  --M_values 3 4 5 \
  --k1 3 \
  --base_seed 2
```

---

> [!IMPORTANT]
> - For the two real-world datasets, IRFL and the TCGA cohort, all preprocessing must be completed before training. For the synthetic experiments, the data can optionally be generated on the fly according to the provided recipe in the configurations, however, for the ablations the corresponding datasets for the `--M_values` chosen, should be already present.

> [!NOTE]
> - For the training of the models in the `main_*.py` files all results are by default logged to Weights & Biases.
> - All the checkpoints are written to a `../checkpoints` directory.
