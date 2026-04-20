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

from posthoc.plotting_config import apply_paper_plot_style
from posthoc.honeybee.helper_metrics import HONEYBEE_MODALITIES, _collect_component_features
from src.models.repercent import RePercENT
from src.utils.helpers import set_seed
from training.main_honeybee import _filter_dataset_by_cancer_types
from training.train_jointopt_2m import make_model_jointopt
from training.train_repercent import make_model

apply_paper_plot_style()


DEFAULT_CANCER_PALETTE = [
    "#d60000", "#8c3bff", "#018700", "#00acc6", "#ff7ed1", "#6b004f",
    "#573b00", "#005659", "#15e18c", "#e6a500", "#ffb3b3", "#7a4900",
    "#0000dd", "#bcb6ff", "#bf03b8", "#645200", "#790000", "#0774d8",
    "#729a00", "#00fdcf", "#b57dff", "#004d26",
]


def sanitize_name(value):
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in str(value))


def filter_split_dataset_by_cancer_types(dataset, filter_cancer_types, split_name):
    filtered_dataset = _filter_dataset_by_cancer_types(dataset, filter_cancer_types)
    if filter_cancer_types is not None:
        if len(filtered_dataset) == 0:
            raise ValueError(f"No {split_name} samples found for cancer types: {filter_cancer_types}.")
        print(f"Filtered cancer types {filter_cancer_types}: {len(filtered_dataset)} {split_name} samples")
    return filtered_dataset


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


def load_split_features(args, script_dir, device, filter_cancer_types=None):
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
    split_dataset = filter_split_dataset_by_cancer_types(
        dataset_split[args.split],
        filter_cancer_types,
        args.split,
    )
    loader = DataLoader(split_dataset, batch_size=args.batch_size, shuffle=False)

    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    checkpoint_path = os.path.join(project_root, checkpoints[args.select_seed])

    set_seed(args.base_seed + args.select_seed)
    model = build_model(args.model_type, model_config, data_config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    return _collect_component_features(loader, model, device, modality_order=HONEYBEE_MODALITIES)


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
