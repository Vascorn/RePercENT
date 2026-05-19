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
from posthoc.honeybee.plot_utils import (
    build_color_map,
    compute_centroid_distance_matrix,
    load_split_features,
    sanitize_name,
)
from training.main_honeybee import DEFAULT_FILTER_CANCER_TYPES, _parse_filter_cancer_types

apply_paper_plot_style()


def _build_unique_embeddings(component_features, labels, source_modality, modality_order=None):
    modality_order = modality_order or HONEYBEE_MODALITIES
    if source_modality not in modality_order:
        raise ValueError(f"Unknown modality {source_modality!r}. Available: {modality_order}")

    labels = np.asarray([str(label) for label in labels])

    component_data = []
    for target_modality in modality_order:
        if target_modality == source_modality:
            continue
        component_name = format_honeybee_component_name("U", source_modality, target_modality)
        component_data.append({
            "source_modality": source_modality,
            "target_modality": target_modality,
            "embeddings": component_features[component_name],
            "labels": labels,
        })

    return component_data




def _component_title(component):
    short_source = get_honeybee_modality_short_name(component["source_modality"])
    short_target = get_honeybee_modality_short_name(component["target_modality"])
    return fr"$U_{{ {short_source} \rightarrow {short_target} }}$"

def _plot_unique_umaps(component_data, output_path, random_state, use_palette=False):
    n_components = len(component_data)
    n_cols = min(3, n_components)
    n_rows = int(np.ceil(n_components / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.8 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    all_cancer_types = sorted({label for component in component_data for label in component["labels"]})
    color_map = build_color_map(all_cancer_types, use_palette=use_palette)

    for ax, component in zip(axes, component_data):
        reduced = reduce_d(component["embeddings"], method="tsne", dim=2, random_state=random_state)
        for cancer_type in all_cancer_types:
            mask = component["labels"] == cancer_type
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

        ax.set_title(_component_title(component))
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    for ax in axes[n_components:]:
        ax.axis("off")

    legend_handles = [
        Line2D([], [], linestyle="", marker="o", markersize=6, color=color_map[cancer_type], label=cancer_type)
        for cancer_type in all_cancer_types
    ]
    fig.legend(legend_handles, all_cancer_types, title="Cancer type", loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout(rect=(0.0, 0.0, 0.86, 1.0))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)



def _plot_unique_distance_heatmaps(component_data, output_path, heatmap_vmax, heatmap_vmin):
    n_components = len(component_data)
    n_cols = min(3, n_components)
    n_rows = int(np.ceil(n_components / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.0 * n_cols, 5.4 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    cancer_types = sorted({label for component in component_data for label in component["labels"]})
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

    for ax, component in zip(axes, component_data):
        distances = compute_centroid_distance_matrix(component["embeddings"], component["labels"], cancer_types)
        sns.heatmap(distances, ax=ax, cbar=True, **heatmap_kwargs)
        ax.set_title(_component_title(component))
        ax.set_xlabel("Cancer type")
        ax.set_ylabel("Cancer type")
        ax.tick_params(axis="x", rotation=90, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)

    for ax in axes[n_components:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot UMAPs and centroid-based angular distance heatmaps of HoneyBee unique components for one source modality")
    parser.add_argument("--datasets_path", type=str, default="../../data/honeybee/datasets/", help="Path to the dataset split directory wrt this script")
    parser.add_argument("--model_type", type=str, choices=["repercent", "gmlp", "gru"], default="repercent", help="Model type to visualize")
    parser.add_argument("--wsi_embedding_mode", type=str, choices=["slide", "patch"], default="slide", help="WSI embedding mode used in the saved split")
    parser.add_argument("--split_seed", type=int, default=42, help="Seed of the saved train/test split")
    parser.add_argument("--split", type=str, choices=["train", "test"], default="test", help="Which saved split to visualize")
    parser.add_argument("--select_seed", type=int, default=0, help="Checkpoint seed index to load")
    parser.add_argument("--base_seed", type=int, default=2, help="Base seed used for reproducibility")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for feature extraction")
    parser.add_argument("--modality", type=str, choices=HONEYBEE_MODALITIES, required=True, help="Source modality whose U_ij components will be visualized")
    parser.add_argument("--filter_cancer_types", nargs="+", default=DEFAULT_FILTER_CANCER_TYPES, help="Optional cancer types to keep, e.g. --filter_cancer_types TCGA-BRCA TCGA-LUAD or TCGA-BRCA,TCGA-LUAD. Should match training.")
    parser.add_argument("--output_dir", type=str, default="figures/unique_component_umap", help="Output directory relative to this script")
    parser.add_argument("--use_palette", action="store_true", help="Use an automatically generated seaborn categorical palette")
    parser.add_argument("--heatmap_vmax", type=float, default=float(np.pi), help="Upper bound for the angular-distance heatmap color scale")
    parser.add_argument("--heatmap_vmin", type=float, default=0.0, help="Lower bound for the angular-distance heatmap color scale")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))

    filter_cancer_types = _parse_filter_cancer_types(args.filter_cancer_types)
    component_features, labels = load_split_features(
        args,
        script_dir,
        device,
        filter_cancer_types=filter_cancer_types,
    )
    component_data = _build_unique_embeddings(
        component_features,
        labels,
        args.modality,
        modality_order=HONEYBEE_MODALITIES,
    )

    if not component_data or component_data[0]["embeddings"].shape[0] == 0:
        raise ValueError(f"No samples were found after filtering cancer types: {filter_cancer_types}")

    output_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    suffix = f"{args.model_type}_{sanitize_name(args.modality)}_{args.split}_seed{args.select_seed}"
    umap_output_path = os.path.join(output_dir, f"unique_component_umap_{suffix}.pdf")
    heatmap_output_path = os.path.join(output_dir, f"unique_component_centroid_angular_distances_{suffix}.pdf")

    _plot_unique_umaps(
        component_data,
        output_path=umap_output_path,
        random_state=args.base_seed + args.select_seed,
        use_palette=args.use_palette,
    )
    _plot_unique_distance_heatmaps(component_data, output_path=heatmap_output_path, heatmap_vmax=args.heatmap_vmax, heatmap_vmin=args.heatmap_vmin)

    print(f"Saved {umap_output_path}")
    print(f"Saved {heatmap_output_path}")


if __name__ == "__main__":
    main()
