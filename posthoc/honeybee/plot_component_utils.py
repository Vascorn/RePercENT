import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from torch.utils.data import DataLoader

from posthoc.honeybee.helper_metrics import HONEYBEE_MODALITIES, _collect_component_features
from src.models.repercent import RePercENT
from src.utils.helpers import set_seed
from training.train_jointopt_2m import make_model_jointopt
from training.train_repercent import make_model


DEFAULT_CANCER_PALETTE = [
    "#d60000", "#8c3bff", "#018700", "#00acc6", "#ff7ed1", "#6b004f",
    "#573b00", "#005659", "#15e18c", "#e6a500", "#ffb3b3", "#7a4900",
    "#0000dd", "#bcb6ff", "#bf03b8", "#645200", "#790000", "#0774d8",
    "#729a00", "#00fdcf", "#b57dff", "#004d26",
]


def sanitize_name(value):
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in str(value))


def build_model(model_type, model_config, data_config, device):
    if model_type == "repercent":
        disen_encoders = [
            make_model(model_config, data_config, modality=m + 1, M=data_config["create_data"]["M"])
            for m in range(data_config["create_data"]["M"])
        ]
        return RePercENT(
            M=data_config["create_data"]["M"],
            disenEncoder=disen_encoders,
            disen_mapping=model_config["repercent"]["disen_mapping"],
            vmfkappa=model_config["repercent"]["vmfkappa"],
        ).to(device)

    if model_type in {"gmlp", "gru"}:
        return make_model_jointopt(model_config).to(device)

    raise ValueError(f"Unsupported model type: {model_type}")


def load_split_features(args, script_dir, device):
    data_config_path = os.path.join(script_dir, "../..", "configs", "data", "honeybee_data.yaml")
    with open(data_config_path, "r") as f:
        data_config = yaml.safe_load(f)

    model_config_path = os.path.join(script_dir, "../..", "configs", "model", f"{args.model_type}_honeybee.yaml")
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    analysis_config_path = os.path.join(script_dir, "../..", "configs", "posthoc_analysis", "honeybee.yaml")
    with open(analysis_config_path, "r") as f:
        analysis_config = yaml.safe_load(f)

    checkpoints = analysis_config[args.model_type]["checkpoints"]
    n_seeds = analysis_config["hyperparameters"]["n_seeds"]
    if n_seeds != len(checkpoints):
        raise ValueError(
            f"Number of seeds in hyperparameters ({n_seeds}) does not match checkpoints ({len(checkpoints)})."
        )
    if not 0 <= args.select_seed < n_seeds:
        raise ValueError(f"select_seed should be between 0 and {n_seeds - 1}, but got {args.select_seed}.")

    dataset_split = torch.load(
        os.path.join(script_dir, args.datasets_path, f"dataset_01_{args.wsi_embedding_mode}_split_{args.split_seed}.pt"),
        weights_only=False,
    )
    loader = DataLoader(dataset_split[args.split], batch_size=args.batch_size, shuffle=False)

    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    checkpoint_path = os.path.join(project_root, checkpoints[args.select_seed])

    set_seed(args.base_seed + args.select_seed)
    model = build_model(args.model_type, model_config, data_config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    return _collect_component_features(loader, model, device, modality_order=HONEYBEE_MODALITIES)


def resolve_selected_cancer_types(labels, requested_cancer_types):
    available = sorted(set(str(label) for label in labels))
    if not requested_cancer_types:
        return available

    selected = [str(cancer_type) for cancer_type in requested_cancer_types]
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"Requested cancer types not found in loaded split: {missing}. Available: {available}")
    return selected


def build_color_map(all_cancer_types, use_palette=False):
    if use_palette:
        if len(all_cancer_types) <= 10:
            palette = sns.color_palette("tab10", n_colors=len(all_cancer_types))
        elif len(all_cancer_types) <= 20:
            palette = sns.color_palette("tab20", n_colors=len(all_cancer_types))
        else:
            palette = sns.color_palette("husl", n_colors=len(all_cancer_types))
        return {cancer_type: palette[idx] for idx, cancer_type in enumerate(all_cancer_types)}

    return {
        cancer_type: DEFAULT_CANCER_PALETTE[idx % len(DEFAULT_CANCER_PALETTE)]
        for idx, cancer_type in enumerate(all_cancer_types)
    }


def l2_normalize(x, axis=-1, eps=1e-12):
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norms, eps, None)


def compute_centroid_distance_matrix(embeddings, labels, cancer_types):
    embeddings = l2_normalize(embeddings, axis=1)
    centroids = []
    for cancer_type in cancer_types:
        mask = labels == cancer_type
        if not np.any(mask):
            raise ValueError(f"No samples available for cancer type {cancer_type} in centroid computation.")
        mean_direction = embeddings[mask].mean(axis=0, keepdims=True)
        centroids.append(l2_normalize(mean_direction, axis=1)[0])

    centroids = np.stack(centroids, axis=0)
    cosine_sim = np.clip(centroids @ centroids.T, -1.0, 1.0)
    return np.arccos(cosine_sim)


def build_heatmap_kwargs(cancer_types, vmax=np.pi / 2.0):
    return {
        "cmap": "coolwarm",
        "square": True,
        "xticklabels": cancer_types,
        "yticklabels": cancer_types,
        "linewidths": 0.2,
        "linecolor": "white",
        "vmin": 0.0,
        "vmax": vmax,
    }


def _circle_distance_matrix(angles):
    deltas = np.abs(angles[:, None] - angles[None, :])
    return np.minimum(deltas, 2.0 * np.pi - deltas)



def _best_fit_circle_angles(distance_matrix, n_restarts=24, n_iters=400, step=0.15):
    n = distance_matrix.shape[0]
    if n == 1:
        return np.asarray([0.0], dtype=np.float64), np.asarray([0], dtype=np.int64)
    if n == 2:
        return np.asarray([0.0, float(distance_matrix[0, 1])], dtype=np.float64), np.asarray([0, 1], dtype=np.int64)

    def objective(angles):
        fitted = _circle_distance_matrix(angles)
        residual = fitted - distance_matrix
        return float(np.mean(residual ** 2))

    base = np.linspace(0.0, 2.0 * np.pi, num=n, endpoint=False)
    best_angles = base.copy()
    best_loss = objective(best_angles)
    rng = np.random.default_rng(0)

    for _ in range(n_restarts):
        angles = np.mod(base + rng.uniform(-np.pi / n, np.pi / n, size=n), 2.0 * np.pi)
        current_step = step
        current_loss = objective(angles)

        for _ in range(n_iters):
            improved = False
            for idx in range(1, n):
                original = angles[idx]
                local_best = current_loss
                local_angle = original
                for delta in (-current_step, current_step):
                    candidate = angles.copy()
                    candidate[idx] = np.mod(original + delta, 2.0 * np.pi)
                    loss = objective(candidate)
                    if loss < local_best:
                        local_best = loss
                        local_angle = candidate[idx]
                if local_best < current_loss:
                    angles[idx] = local_angle
                    current_loss = local_best
                    improved = True

            if not improved:
                current_step *= 0.85
                if current_step < 1e-4:
                    break

        if current_loss < best_loss:
            best_loss = current_loss
            best_angles = angles.copy()

    order = np.argsort(best_angles)
    return best_angles[order], order


def plot_circular_distance_layouts(items, output_path, title_fn, matrix_fn, color_map):
    n_items = len(items)
    n_cols = min(3, n_items)
    n_rows = int(np.ceil(n_items / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.2 * n_cols, 5.8 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, item in zip(axes, items):
        labels = sorted(set(item["labels"]))
        distances = matrix_fn(item, labels)

        if len(labels) == 1:
            ordered_labels = labels
            angles = np.asarray([0.0], dtype=np.float64)
        else:
            angles, order = _best_fit_circle_angles(distances)
            ordered_labels = [labels[idx] for idx in order]

        theta = np.linspace(0.0, 2.0 * np.pi, 400)
        ax.plot(np.cos(theta), np.sin(theta), linestyle="--", linewidth=1.0, color="#888888", alpha=0.8)

        for label, angle in zip(ordered_labels, angles):
            x = np.cos(angle)
            y = np.sin(angle)
            lx = 1.14 * x
            ly = 1.14 * y
            ax.scatter(x, y, s=70, color=color_map[label], edgecolors="black", linewidths=0.7, zorder=3)
            ax.text(lx, ly, label, ha="center", va="center", fontsize=8)

        ax.set_title(f"{title_fn(item)}\nBest-Fit Circular Layout")
        ax.set_aspect("equal")
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.35)
        ax.set_axis_off()

    for ax in axes[n_items:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
