# Training Pipelines

This directory contains the training pipelines for __RePercENT__ and the __JointOpt__ variants. For training hyperparameters, see `../configs/training`; for model architecture settings, see `../configs/model`.

## Directory Map

The directory's logic can be split into the following main components:

* `main_*.py` scripts are executable entry points for full training runs.
* `train_*.py` scripts provide model factories, training loops, test loops, and evaluation helpers used by the entry points.
* `log_data.py` provides helper functions for logging datasets, model configs, and checkpoints to Weights & Biases.

### Entry points

| File | Description |
| --- | --- |
| `main.py` | Main synthetic-data training entry point for RePercENT and JointOpt baselines. The script currently uses `M = 2` internally and loads the matching synthetic data, model, and training configs. The script training the model according to the loaded `../configs` recipe and evaluated the final model on the test set, using linear probing. |
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
| `log_data.py` | Weights & Biases logging utilities for model details, checkpoints, and dataset artifacts. |

## How to use: 

### Synthetic Experiments

![Dataset: Synthetic](https://img.shields.io/badge/dataset-Synthetic-pink)
![Script](https://img.shields.io/badge/script-main.py-lightgrey)


To train on the synthetic dataset, use the main synthetic entry point:

```bash
python training/main.py \
  --load_data True \ # whether to load a saved synthetic dataset
  --model_type repercent \ # switch to jointopt for the gmlp, gru, mlp baselines
  --k1 3 \ # set number of train/val/test splits
  --k2 2 \ # number of random seeds per split
  --base_seed 2 # base seed for reproducibility
``` 

---

### IRFL experiments

![Dataset: IRFL](https://img.shields.io/badge/dataset-IRFL-blue)
![Script](https://img.shields.io/badge/script-main__irfl.py-lightgrey)

To train on IRFL, use the preprocessed IRFL tensors and select the model variant:

```bash
python training/main_irfl.py \
  --datasets_path ../data/irfl/datasets/ \ # path to preprocessed IRFL tensors
  --model_type repercent \ # switch to gmlp or gru for the JointOpt baselines
  --n_seeds 5 \ # number of random seeds
  --base_seed 2 \ # base seed for reproducibility
  --comp_mod 1 # comparison modality for image retrieval evaluation at the end of the training. 1 is the default for the Caption, 2 stands for Definition. The latter is valid only if all three modalities are used (Image, Caption, Definition) 
```

---

### HONeYBEE/ TCGA cohort

![Dataset: HONeYBEE](https://img.shields.io/badge/dataset-HONeYBEE-green)
![Script](https://img.shields.io/badge/script-main__honeybee.py-lightgrey)

To train on the Honeybee/TCGA cohort, use the Honeybee entry point and select the model variant:

```bash
python training/main_honeybee.py \
  --datasets_path ../data/honeybee/datasets/ \ # path to preprocessed Honeybee/TCGA tensors
  --load_test_split True \ # switch to False for generating a new train/ test split
  --model_type repercent \ # switch to gmlp or gru for the JointOpt baselines
  --n_seeds 5 \ # number of random seeds
  --base_seed 2 \ # base seed for model initialization
  --split_seed 42 \ # seed for stratified train/test split
  --wsi_embedding_mode slide \ # use slide-level WSI embeddings
  --filter_cancer_types TCGA-BRCA TCGA-COAD TCGA-GBM TCGA-HNSC TCGA-KIRC TCGA-LGG TCGA-LUAD TCGA-LUSC TCGA-OV TCGA-PRAD # filter out specific cancer type for training
```

---
### Alpha ablations

![Ablation](https://img.shields.io/badge/experiment-alpha_ablation-purple)
![Script](https://img.shields.io/badge/script-main__ablations__alpha.py-lightgrey)

To run the RePercENT alpha ablations across modality counts, provide the alpha values and modality counts explicitly:

```bash
python training/main_ablations_alpha.py \
  --model_type repercent \ # switch to jointopt for baseline ablations
  --alpha_values 0.01 0.1 1.0 10.0 100.0 \ # alpha values to sweep
  --M_values 3 4 5 \ # modality counts to evaluate
  --k1 3 \ # number of train/val/test splits
  --base_seed 2 # base seed for reproducibility
```

---

> [!IMPORTANT]
> - For the two real-world datasets, IRFL and the TCGA cohort, all preprocessing must be completed before training. For the synthetic experiments, the data can optionally be generated on the fly; however, to reproduce the reported results, use the pre-generated datasets in `../data/repercent_synthetic/` and select the appropriate `dataset2{M}` directory, where `{M}` is the number of modalities: 2, 3, 4, or 5.

> [!NOTE]
> - For the training of the models in the `main_*.py` files all results are by default logged to Weights & Biases.
> - All the checkpoints are written to a `../checkpoints` directory.
