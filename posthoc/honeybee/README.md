# HONeYBEE Posthoc Analysis

Posthoc scripts for the Honeybee experiments: cancer-type probe metrics, simple
raw-embedding baselines, RePercENT-vs-baseline comparisons, UMAP visualizations,
and missing-modality robustness.

Run commands from this directory:

```bash
cd posthoc/honeybee
```

## Inputs

The scripts expect fixed Honeybee split files in `../../data/honeybee/datasets/`,
such as `dataset_01_slide_split_42.pt`. RePercENT checkpoint paths are read from
`../../configs/posthoc_analysis/honeybee.yaml`.

## Metric Summaries

Generate RePercENT component probe metrics:

```bash
python calc_metrics.py --model_type repercent --wsi_embedding_mode slide --split_seed 42
```

Generate simple raw-embedding baselines:

```bash
python simple_baselines.py --wsi_embedding_mode slide --split_seed 42
```

Both scripts save local CSV summaries to:

```text
summary_reports/cancer_type_component_summary/
```

The CSV format is long-form: `component, eval, mean, std`.

Compare RePercENT against the simple baselines:

```bash
python compare_cancer_type_runs.py
```

By default this reads:

```text
summary_reports/cancer_type_component_summary/repercent_cancer_type_component_summary.csv
summary_reports/cancer_type_component_summary/simple_baselines_cancer_type_component_summary.csv
```

and writes the results to `figures/cancer_type_component_summary/`.

## UMAP Figures

For generating different umap embeddings for the unique and shared components of the tcga cohort, you may run:

```bash
python plot_raw_embedding_umap_by_cancer_type.py --split test --wsi_embedding_mode slide
python plot_unique_component_umap_by_cancer_type.py --modality clinical_qwen --split test
python plot_shared_pair_umap_by_cancer_type.py --split test
```

Default outputs are written under `figures/raw_embedding_umap/`,
`figures/unique_component_umap/`, and `figures/shared_pair_umap/`.

## Missingness Analysis

```bash
python missingness_eval.py --metric macro_f1
python missingness_eval.py --metric balanced_accuracy
```

Existing summary CSVs are reused to regenerate figures. Missingness CSVs are
stored in `summary_reports/missingess_summary/`, and panel PDFs are stored in
`figures/missingness/`.

## Notes

- `calc_metrics.py` and `simple_baselines.py` save local CSVs by default; W&B
  logging is optional via `--log_to_wandb True`.
- `missingness_eval.py` does not contact W&B when regenerating plots from an
  existing summary CSV. Optional logging to W&B via `--log_to_wandb True`.
- The default cancer-type subset is BRCA, COAD, GBM, HNSC, KIRC, LGG, LUAD,
  LUSC, OV, and PRAD.
