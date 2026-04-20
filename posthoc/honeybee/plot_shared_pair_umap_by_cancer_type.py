import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import numpy as np
import torch

from posthoc.plotting_config import apply_paper_plot_style
from posthoc.irfl.helper_vis import reduce_d
from posthoc.honeybee.helper_metrics import (
    HONEYBEE_MODALITIES,
    format_honeybee_component_name,
    get_honeybee_modality_short_name,
)
from posthoc.honeybee.plot_component_utils import (
    build_color_map,
    compute_centroid_distance_matrix,
    load_split_features,
)
from training.main_honeybee import DEFAULT_FILTER_CANCER_TYPES, _parse_filter_cancer_types

apply_paper_plot_style()


def _build_pair_embeddings(component_features, labels, modality_order=None):
    modality_order = modality_order or HONEYBEE_MODALITIES
    labels = np.asarray([str(label) for label in labels])

    pair_data = []
    for i in range(len(modality_order)):
        for j in range(i + 1, len(modality_order)):
            modality_a = modality_order[i]
            modality_b = modality_order[j]
            forward_name = format_honeybee_component_name("S", modality_a, modality_b)
            reverse_name = format_honeybee_component_name("S", modality_b, modality_a)
            shared_forward = component_features[forward_name]
            shared_reverse = component_features[reverse_name]
            pair_data.append({
                "modality_a": modality_a,
                "modality_b": modality_b,
                "shared_forward": shared_forward,
                "shared_reverse": shared_reverse,
                "labels": labels,
            })

    return pair_data


def _select_shared_embeddings(pair, shared_direction):
    if shared_direction == "forward":
        return pair["shared_forward"]
    if shared_direction == "reverse":
        return pair["shared_reverse"]
    if shared_direction == "both":
        if pair["shared_forward"].shape != pair["shared_reverse"].shape:
            raise ValueError(
                "Forward and reverse shared embeddings must have the same shape "
                f"to average them, got {pair['shared_forward'].shape} and {pair['shared_reverse'].shape}."
            )
        return 0.5 * (pair["shared_forward"] + pair["shared_reverse"])

    raise ValueError(f"Unsupported shared direction: {shared_direction}")


def _pair_direction_title(pair, shared_direction):
    short_a = get_honeybee_modality_short_name(pair["modality_a"])
    short_b = get_honeybee_modality_short_name(pair["modality_b"])

    if shared_direction == "both":
        return fr"$S_{{ {short_a} \leftrightarrow {short_b} }}$"
    if shared_direction == "forward":
        return fr"$S_{{ {short_a} \rightarrow {short_b} }}$"
    if shared_direction == "reverse":
        return fr"$S_{{ {short_b} \rightarrow {short_a} }}$"

    raise ValueError(f"Unsupported shared direction: {shared_direction}")


def _build_pair_umap_panels(pair_data, shared_direction):
    panels = []
    for pair in pair_data:
        panels.append({
            "embeddings": _select_shared_embeddings(pair, shared_direction),
            "labels": pair["labels"],
            "title": _pair_direction_title(pair, shared_direction),
        })

    return panels


def _plot_pair_umaps(pair_data, output_path, random_state, use_palette=False, shared_direction="both"):
    panels = _build_pair_umap_panels(pair_data, shared_direction)
    n_panels = len(panels)
    n_cols = 3
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.8 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    all_cancer_types = sorted({label for pair in pair_data for label in pair["labels"]})
    color_map = build_color_map(all_cancer_types, use_palette=use_palette)

    for ax, panel in zip(axes, panels):
        reduced = reduce_d(panel["embeddings"], method="tsne", dim=2, random_state=random_state)
        for cancer_type in all_cancer_types:
            mask = panel["labels"] == cancer_type
            if not np.any(mask):
                continue
            ax.scatter(
                reduced[mask, 0],
                reduced[mask, 1],
                s=16,
                alpha=0.8,
                color=color_map[cancer_type],
                label=cancer_type,
            )

        ax.set_title(panel["title"])
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    for ax in axes[n_panels:]:
        ax.axis("off")

    legend_handles = [
        Line2D([], [], linestyle="", marker="o", markersize=6, color=color_map[cancer_type], label=cancer_type)
        for cancer_type in all_cancer_types
    ]
    fig.legend(legend_handles, all_cancer_types, title="Cancer type", loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout(rect=(0.0, 0.0, 0.86, 1.0))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _compute_pair_distance_matrix(pair, cancer_types, shared_direction):
    if shared_direction == "both":
        forward_distances = compute_centroid_distance_matrix(
            pair["shared_forward"],
            pair["labels"],
            cancer_types,
        )
        reverse_distances = compute_centroid_distance_matrix(
            pair["shared_reverse"],
            pair["labels"],
            cancer_types,
        )
        return 0.5 * (forward_distances + reverse_distances)

    return compute_centroid_distance_matrix(
        _select_shared_embeddings(pair, shared_direction),
        pair["labels"],
        cancer_types,
    )


def _plot_pair_distance_heatmaps(pair_data, output_path, heatmap_vmax, heatmap_vmin, shared_direction):
    n_pairs = len(pair_data)
    n_cols = 3
    n_rows = int(np.ceil(n_pairs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.0 * n_cols, 5.4 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    cancer_types = sorted({label for pair in pair_data for label in pair["labels"]})
    heatmap_kwargs = {
        "cmap": "coolwarm",
        "square": True,
        "xticklabels": cancer_types,
        "yticklabels": cancer_types,
        "linewidths": 0.2,
        "linecolor": "white",
        "vmin": heatmap_vmin,
        "vmax": heatmap_vmax
    }

    for ax, pair in zip(axes, pair_data):
        distances = _compute_pair_distance_matrix(pair, cancer_types, shared_direction)
        sns.heatmap(distances, ax=ax, cbar=True, **heatmap_kwargs)
        ax.set_title(_pair_direction_title(pair, shared_direction))
        ax.set_xlabel("Cancer type")
        ax.set_ylabel("Cancer type")
        ax.tick_params(axis="x", rotation=90, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)

    for ax in axes[n_pairs:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot UMAPs and centroid-based angular distance heatmaps of shared HoneyBee pair components")
    parser.add_argument("--datasets_path", type=str, default="../../data/honeybee/datasets/", help="Path to the dataset split directory wrt this script")
    parser.add_argument("--model_type", type=str, choices=["repercent", "gmlp", "gru"], default="repercent", help="Model type to visualize")
    parser.add_argument("--wsi_embedding_mode", type=str, choices=["slide", "patch"], default="slide", help="WSI embedding mode used in the saved split")
    parser.add_argument("--split_seed", type=int, default=42, help="Seed of the saved train/test split")
    parser.add_argument("--split", type=str, choices=["train", "test"], default="test", help="Which saved split to visualize")
    parser.add_argument("--select_seed", type=int, default=0, help="Checkpoint seed index to load")
    parser.add_argument("--base_seed", type=int, default=2, help="Base seed used for reproducibility")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for feature extraction")
    parser.add_argument("--filter_cancer_types", nargs="+", default=DEFAULT_FILTER_CANCER_TYPES, help="Optional cancer types to keep, e.g. --filter_cancer_types TCGA-BRCA TCGA-LUAD or TCGA-BRCA,TCGA-LUAD. Should match training.")
    parser.add_argument("--output_dir", type=str, default="figures/shared_pair_umap", help="Output directory relative to this script")
    parser.add_argument("--use_palette", action="store_true", help="Use an automatically generated seaborn categorical palette")
    parser.add_argument("--shared_direction", type=str, choices=["forward", "reverse", "both"], default="both", help="Which shared direction to plot: forward uses Sij, reverse uses Sji, both averages Sij and Sji.")
    parser.add_argument("--heatmap_vmax", type=float, default=float(np.pi), help="Upper bound for the angular-distance heatmap color scale")
    parser.add_argument("--heatmap_vmin", type=float, default=0.0, help="Lower bound for the angular-distance heatmap color scale")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))

    filter_cancer_types = _parse_filter_cancer_types(args.filter_cancer_types)
    component_features, labels, _ = load_split_features(
        args,
        script_dir,
        device,
        filter_cancer_types=filter_cancer_types,
    )
    pair_data = _build_pair_embeddings(component_features, labels, modality_order=HONEYBEE_MODALITIES)

    if not pair_data or pair_data[0]["shared_forward"].shape[0] == 0:
        raise ValueError(f"No samples were found after filtering cancer types: {filter_cancer_types}")

    output_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    suffix = f"{args.model_type}_{args.split}_seed{args.select_seed}"
    umap_output_path = os.path.join(output_dir, f"shared_pair_umap_{args.shared_direction}_{suffix}.pdf")
    heatmap_output_path = os.path.join(output_dir, f"shared_pair_centroid_angular_distances_{args.shared_direction}_{suffix}.pdf")

    _plot_pair_umaps(
        pair_data,
        output_path=umap_output_path,
        random_state=args.base_seed + args.select_seed,
        use_palette=args.use_palette,
        shared_direction=args.shared_direction,
    )
    _plot_pair_distance_heatmaps(
        pair_data,
        output_path=heatmap_output_path,
        heatmap_vmax=args.heatmap_vmax,
        heatmap_vmin=args.heatmap_vmin,
        shared_direction=args.shared_direction,
    )

    print(f"Saved {umap_output_path}")
    print(f"Saved {heatmap_output_path}")


if __name__ == "__main__":
    main()
