import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from posthoc.plotting_config import apply_paper_plot_style
from posthoc.irfl.helper_vis import reduce_d
from posthoc.honeybee.helper_metrics import HONEYBEE_MODALITIES, get_honeybee_modality_short_name
from posthoc.honeybee.plot_component_utils import (
    build_color_map,
    filter_split_dataset_by_cancer_types,
    sanitize_name,
)
from training.main_honeybee import DEFAULT_FILTER_CANCER_TYPES, _parse_filter_cancer_types

apply_paper_plot_style()


def _masked_mean(embeddings, mask):
    embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
    mask = torch.as_tensor(mask, dtype=torch.bool)

    while mask.ndim < embeddings.ndim:
        mask = mask.unsqueeze(-1)

    masked_embeddings = embeddings * mask.to(dtype=embeddings.dtype)
    valid_count = mask.to(dtype=embeddings.dtype).sum(dim=0).clamp_min(1.0)
    return masked_embeddings.sum(dim=0) / valid_count


def _fuse_modality_embeddings(embeddings, pad_mask):
    embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
    pad_mask = torch.as_tensor(pad_mask, dtype=torch.bool)

    if embeddings.ndim in {2, 3}:
        return _masked_mean(embeddings, pad_mask)

    if embeddings.ndim == 4:
        slide_embeddings = []
        slide_mask = pad_mask.any(dim=-1)
        for slide_idx in range(embeddings.shape[0]):
            if not bool(slide_mask[slide_idx]):
                continue
            slide_embeddings.append(_masked_mean(embeddings[slide_idx], pad_mask[slide_idx]))

        if not slide_embeddings:
            return torch.zeros(embeddings.shape[-1], dtype=embeddings.dtype)
        return torch.stack(slide_embeddings, dim=0).mean(dim=0)

    raise ValueError(f"Unsupported embedding shape for fusion: {tuple(embeddings.shape)}")


def _extract_raw_modality_features(dataset, modality_order=None):
    modality_order = modality_order or HONEYBEE_MODALITIES
    feature_store = {modality: [] for modality in modality_order}
    labels = []

    for sample_idx in range(len(dataset)):
        sample = dataset[sample_idx]

        for modality in modality_order:
            embeddings, _, pad_mask, has_data = sample[modality]
            if not has_data:
                raise ValueError(f"Missing modality {modality} for sample {sample_idx}.")

            fused_feature = _fuse_modality_embeddings(embeddings, pad_mask)
            feature_store[modality].append(fused_feature.numpy())

        labels.append(str(sample["cancer_type"]))

    feature_store = {
        name: np.stack(features, axis=0)
        for name, features in feature_store.items()
    }
    return feature_store, np.asarray(labels)


def _build_modality_data(raw_features, labels, modality_order=None):
    modality_order = modality_order or HONEYBEE_MODALITIES
    labels = np.asarray([str(label) for label in labels])

    return [
        {
            "modality": modality,
            "embeddings": raw_features[modality],
            "labels": labels,
        }
        for modality in modality_order
    ]


def _plot_raw_umaps(modality_data, output_path, random_state, use_palette=False):
    n_modalities = len(modality_data)
    n_cols = 2
    n_rows = int(np.ceil(n_modalities / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.8 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    all_cancer_types = sorted({label for item in modality_data for label in item["labels"]})
    color_map = build_color_map(all_cancer_types, use_palette=use_palette)

    for ax, item in zip(axes, modality_data):
        reduced = reduce_d(item["embeddings"], method="tsne", dim=2, random_state=random_state)
        for cancer_type in all_cancer_types:
            mask = item["labels"] == cancer_type
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

        ax.set_title(get_honeybee_modality_short_name(item["modality"]))
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    for ax in axes[n_modalities:]:
        ax.axis("off")

    legend_handles = [
        Line2D([], [], linestyle="", marker="o", markersize=6, color=color_map[cancer_type], label=cancer_type)
        for cancer_type in all_cancer_types
    ]
    fig.legend(legend_handles, all_cancer_types, title="Cancer type", loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout(rect=(0.0, 0.0, 0.86, 1.0))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot UMAPs of raw HoneyBee modality embeddings by cancer type")
    parser.add_argument("--datasets_path", type=str, default="../../data/honeybee/datasets/", help="Path to the dataset split directory wrt this script")
    parser.add_argument("--wsi_embedding_mode", type=str, choices=["slide", "patch"], default="slide", help="WSI embedding mode used in the saved split")
    parser.add_argument("--split_seed", type=int, default=42, help="Seed of the saved train/test split")
    parser.add_argument("--split", type=str, choices=["train", "test"], default="test", help="Which saved split to visualize")
    parser.add_argument("--base_seed", type=int, default=2, help="Base seed used for reproducibility")
    parser.add_argument("--filter_cancer_types", nargs="+", default=DEFAULT_FILTER_CANCER_TYPES, help="Optional cancer types to keep, e.g. --filter_cancer_types TCGA-BRCA TCGA-LUAD or TCGA-BRCA,TCGA-LUAD. Should match training.")
    parser.add_argument("--output_dir", type=str, default="figures/raw_embedding_umap", help="Output directory relative to this script")
    parser.add_argument("--use_palette", action="store_true", help="Use an automatically generated seaborn categorical palette")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_split = torch.load(
        os.path.join(script_dir, args.datasets_path, f"dataset_01_{args.wsi_embedding_mode}_split_{args.split_seed}.pt"),
        weights_only=False,
    )
    filter_cancer_types = _parse_filter_cancer_types(args.filter_cancer_types)
    split_dataset = filter_split_dataset_by_cancer_types(
        dataset_split[args.split],
        filter_cancer_types,
        args.split,
    )

    raw_features, labels = _extract_raw_modality_features(split_dataset, modality_order=HONEYBEE_MODALITIES)
    modality_data = _build_modality_data(raw_features, labels, modality_order=HONEYBEE_MODALITIES)

    if not modality_data or modality_data[0]["embeddings"].shape[0] == 0:
        raise ValueError(f"No samples were found after filtering cancer types: {filter_cancer_types}")

    output_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    suffix = f"{args.split}_{sanitize_name(args.wsi_embedding_mode)}_seed{args.split_seed}"
    umap_output_path = os.path.join(output_dir, f"raw_embedding_umap_{suffix}.pdf")

    _plot_raw_umaps(
        modality_data,
        output_path=umap_output_path,
        random_state=args.base_seed,
        use_palette=args.use_palette,
    )

    print(f"Saved {umap_output_path}")


if __name__ == "__main__":
    main()
