# `IRFL` Posthoc Analysis
![Dataset](https://img.shields.io/badge/dataset-IRFL-blue)
![Task](https://img.shields.io/badge/task-Detection-purple)

Posthoc scripts for the IRFL experiments, inluding:
1. the IRFL detection task
2. embedding visualizations of the unique and shared components

Run commands from this directory:

```bash
cd posthoc/irfl
```

## Inputs

The scripts expect fixed IRFL split files in `../../data/irfl/datasets/`,
including `IRFL_test_tensors_2.pt` and the corresponding augmentations `IRFL_test_tensors_aug__2.pt` for creating the test data loader. Checkpoint paths for the different models are read from
`../../configs/posthoc_analysis/irfl.yaml`.

## Metric Summaries

Generate the evaluation scores, including accuracy, margin of the correct image compared to the distractors, MRR, using:

```bash
python calc_metrics.py \
    --model_type repercent \
    --component shared \
    --base_seed 2
```

By default this loads the corresponding pre-trained model from a path specified in `../../configs/posthoc_analysis/irfl_3m.yaml`.

## Visualizations

For generating the different unique and shared component embeddings, use e.g.:

```bash
python visualize_embeddings.py \
    --model_type repercent \
    --component shared \
    --select_seed 1
```

Default figures are written under `figures/embeddings/`.