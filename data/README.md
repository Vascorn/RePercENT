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

