# `Synthetic` data Posthoc Analysis
![Dataset](https://img.shields.io/badge/dataset-Synthetic-pink)
![Task](https://img.shields.io/badge/task-Linear%20Probe-purple)

Posthoc script for the synthetic experiments, including for each examined model:
1. the linear probe performance among all extracted latent representations
2. number of FLOPs, parameters and inference latency

Run commands from this directory:

```bash
cd posthoc/synthetic
```

## Inputs

The scripts expect fixed IRFL split files in `../../data/repercent_syntetic/dataset2{M}`, for creating the exact train and test data loaders. Checkpoint paths for the different trained models are read from
`../../configs/posthoc_analysis/synthetic_{M}m.yaml`.

## Metric Summaries

Generate the evaluation scores, including accuracy, margin of the correct image compared to the distractors, MRR, using:

```bash
python calc_metrics.py \
    --model_type repercent \
    --M 2 \
    --base_seed 2 \
    --k1 3 \
    --k2 2
```

By default this loads the corresponding pre-trained models from a path specified in `../../configs/posthoc_analysis/synthetic_{M}m.yaml`. The parameters `--k1` and `--k2` represent the separte splits and seeds for each split the models are trained on, and should match the training protocol. The final confusion matrices across all runs are saved in `/figures/confusion_matrices`.


> [!NOTE]
> Here `{M}` is replaced by the value passed to `--M`, which denotes the number of modalities used in the experiment.