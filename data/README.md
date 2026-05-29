# Data Directory

This directory documents the expected local data layout for RePercENT experiments.
The dataset files themselves are intentionally not tracked in git and are excluded
from Docker image builds.

Place local datasets under the following paths:

```text
data/
├── honeybee/
│   └── datasets/
├── irfl/
│   ├── images/
│   └── datasets/
└── repercent_synthetic/
    ├── dataset22/
    ├── dataset23/
    ├── dataset24/
    └── dataset25/
```

Notes:

- `data/honeybee/` is used by `training/main_honeybee.py`.
- `data/irfl/` is used by `training/main_irfl.py`.
- `data/repercent_synthetic/` is used by synthetic training and posthoc scripts.

## IRFL Preprocessing

The IRFL preprocessing notebook has a script equivalent at
`src/utils/irfl_preprocess.py`. The script downloads the IRFL CSV tables from
Hugging Face, builds the figurative-only train split and detection-task test
split, removes train images that overlap with test answers, and writes the CSV
files under `data/irfl/datasets/`.

The script also downloads `IRFL_images.zip` when `data/irfl/images/` is missing
or empty, then flattens the archive into:

```text
data/irfl/images/<image_id>.jpeg
```

To regenerate only the CSV files, run:

```bash
docker compose run --rm repercent python src/utils/irfl_preprocess.py --csv-only
```

To generate the full tensor files, run:

```bash
docker compose run --rm repercent python src/utils/irfl_preprocess.py
```

This extracts OpenCLIP text token embeddings and ViT image patch embeddings for
the original and augmented IRFL views. On a Linux machine with an NVIDIA GPU, use
the GPU Compose service instead:

```bash
docker compose --profile gpu run --rm repercent-gpu python src/utils/irfl_preprocess.py
```

The default outputs are:

```text
data/irfl/datasets/IRFL_train_tensors_2.pt
data/irfl/datasets/IRFL_test_tensors_2.pt
data/irfl/datasets/IRFL_train_tensors_aug_2.pt
data/irfl/datasets/IRFL_test_tensors_aug_2.pt
```

The intermediate CSV files include:

```text
data/irfl/datasets/IRFL_train_dataset_2.csv
data/irfl/datasets/IRFL_test_detect_dataset_2.csv
data/irfl/datasets/IRFL_complete_datasets_w_all_defs.csv
data/irfl/datasets/IRFL_complete_datasets_full_w_all_defs.csv
```

## HONeYBEE/TCGA Preprocessing

Honeybee/TCGA preprocessing is handled by `src/utils/honeybee_dataset.py`. It
loads the TCGA embeddings from the HONeYBEE Hugging Face dataset, aligns patients
across the configured modalities, pads variable-length modality inputs, and
writes a local `MultimodalTCGA` tensor dataset.

For the slide-level WSI representation used by the training scripts, run:

```bash
docker compose run --rm repercent python src/utils/honeybee_dataset.py \
  --data_save_path ./data/honeybee/datasets/ \
  --wsi_embedding_mode slide
```

The expected output is:

```text
data/honeybee/datasets/dataset_01_slide.pt
```

`training/main_honeybee.py` loads a fixed stratified split by default. The split
file is expected at:

```text
data/honeybee/datasets/dataset_01_slide_split_42.pt
```

If you create a new split through `training/main_honeybee.py`, keep the same
`--wsi_embedding_mode` and `--split_seed` values for training and posthoc
analysis.
