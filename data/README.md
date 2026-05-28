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
- Generated checkpoints should be written to `checkpoints/`, which is also ignored.

## IRFL Preprocessing

The IRFL preprocessing notebook has a script equivalent at
`src/utils/irfl_preprocess.py`. It downloads the IRFL CSV tables from Hugging
Face, recreates the train/test split, extracts CLIP token embeddings, and writes
the tensor files expected by `training/main_irfl.py`.

Place the IRFL images as JPEG files under:

```text
data/irfl/images/<image_id>.jpeg
```

Then run the preprocessing from the Docker environment:

```bash
docker compose run --rm repercent python src/utils/irfl_preprocess.py --csv-only
```

Use `--csv-only` when you only want to regenerate the CSV split. To generate the
full tensor files, run:

```bash
docker compose run --rm repercent python src/utils/irfl_preprocess.py
```

On a Linux machine with an NVIDIA GPU, use the GPU Compose service instead:

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
