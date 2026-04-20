import argparse
import os
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from posthoc.plotting_config import apply_paper_plot_style
from posthoc.honeybee.helper_metrics import (
    HONEYBEE_MODALITIES,
    get_honeybee_modality_short_name,
    test_fwd_only,
)
from posthoc.irfl.helper_vis import reduce_d
from src.models.repercent import RePercENT
from src.utils.helpers import set_seed
from training.main_honeybee import (
    DEFAULT_FILTER_CANCER_TYPES,
    _filter_dataset_by_cancer_types,
    _parse_filter_cancer_types,
)
from training.train_jointopt_2m import make_model_jointopt
from training.train_repercent import make_model

apply_paper_plot_style()


PAIR_COLORS = {
    "unique_forward": "skyblue",
    "shared_forward": "dodgerblue",
    "unique_reverse": "lightcoral",
    "shared_reverse": "red",
}


def _sanitize_name(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def _build_modality_aliases(modality_names):
    aliases = {}
    used_codes = set()

    for modality_name in modality_names:
        short_name = get_honeybee_modality_short_name(modality_name)
        base_code = short_name[0].lower()
        code = base_code
        suffix = 2
        while code in used_codes:
            code = f"{base_code}{suffix}"
            suffix += 1
        aliases[modality_name] = code
        used_codes.add(code)

    return aliases


def _available_cancer_types(dataset, filter_cancer_types):
    available = sorted({str(dataset[idx]["cancer_type"]) for idx in range(len(dataset))})
    if filter_cancer_types is None:
        return available

    available_set = set(available)
    return [
        cancer_type
        for cancer_type in dict.fromkeys(str(cancer_type) for cancer_type in filter_cancer_types)
        if cancer_type in available_set
    ]


def _collect_pair_embeddings(loader, model, device, cancer_type, modality_order=None):
    modality_order = modality_order or HONEYBEE_MODALITIES
    pair_embeddings = {}

    for i in range(len(modality_order)):
        for j in range(i + 1, len(modality_order)):
            pair_embeddings[(i, j)] = {
                "unique_forward": [],
                "shared_forward": [],
                "unique_reverse": [],
                "shared_reverse": [],
            }

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch_labels = np.asarray([str(label) for label in batch["cancer_type"]])
            keep_mask = batch_labels == cancer_type
            if not np.any(keep_mask):
                continue

            outputs = test_fwd_only(batch, model, device, modality_order=modality_order)
            U = F.normalize(outputs["U"].detach().cpu(), dim=-1)
            S_view = F.normalize(outputs["S_view"].detach().cpu(), dim=-1)
            keep_mask_t = torch.as_tensor(keep_mask, dtype=torch.bool)

            for i in range(len(modality_order)):
                for j in range(i + 1, len(modality_order)):
                    pair_embeddings[(i, j)]["unique_forward"].append(U[keep_mask_t, i, j, :].numpy())
                    pair_embeddings[(i, j)]["shared_forward"].append(S_view[keep_mask_t, i, j, :].numpy())
                    pair_embeddings[(i, j)]["unique_reverse"].append(U[keep_mask_t, j, i, :].numpy())
                    pair_embeddings[(i, j)]["shared_reverse"].append(S_view[keep_mask_t, j, i, :].numpy())

    out = {}
    for pair, components in pair_embeddings.items():
        out[pair] = {}
        for name, chunks in components.items():
            if not chunks:
                out[pair][name] = np.empty((0, 0), dtype=np.float32)
                continue
            out[pair][name] = np.concatenate(chunks, axis=0)

    return out


def _plot_pair_umap(pair_embeddings, modality_a, modality_b, cancer_type, random_state, fig_path, modality_aliases, add_aliases_legend= False):
    component_order = ["unique_forward", "shared_forward", "unique_reverse", "shared_reverse"]
    modality_a_short = get_honeybee_modality_short_name(modality_a)
    modality_b_short = get_honeybee_modality_short_name(modality_b)
    code_a = modality_aliases[modality_a]
    code_b = modality_aliases[modality_b]
    component_labels = {
        "unique_forward": f"$U_{{{code_a}{code_b}}}$",
        "shared_forward": f"$S_{{{code_a}{code_b}}}$",
        "unique_reverse": f"$U_{{{code_b}{code_a}}}$",
        "shared_reverse": f"$S_{{{code_b}{code_a}}}$",
    }

    component_arrays = [pair_embeddings[name] for name in component_order]
    if any(array.shape[0] == 0 for array in component_arrays):
        raise ValueError(
            f"Not enough embeddings to plot pair {modality_a}-{modality_b} for cancer type {cancer_type}."
        )

    lengths = [array.shape[0] for array in component_arrays]
    reduced = reduce_d(
        np.concatenate(component_arrays, axis=0),
        method="umap",
        dim=2,
        random_state=random_state,
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter_handles = []
    start = 0
    for name, length in zip(component_order, lengths):
        chunk = reduced[start:start + length]
        start += length
        handle = ax.scatter(
            chunk[:, 0],
            chunk[:, 1],
            label=component_labels[name],
            c=PAIR_COLORS[name],
            alpha=0.75,
            s=26,
        )
        scatter_handles.append(handle)

    ax.set_title(f"{cancer_type}: {modality_a_short} vs {modality_b_short}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    component_legend = ax.legend(handles=scatter_handles, title="Components", loc="best")
    ax.add_artist(component_legend)

    if add_aliases_legend:
        alias_handles = [
            Line2D([], [], color="none", label=f"{code} = {get_honeybee_modality_short_name(name)}")
            for name, code in modality_aliases.items()
        ]
        ax.legend(
        handles=alias_handles,
        title="Modality aliases",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        handlelength=0,
        handletextpad=0,
    )

    fig.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _build_model(model_type, model_config, data_config, device):
    match model_type:
        case "repercent":
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
        case "gmlp" | "gru":
            return make_model_jointopt(model_config).to(device)
        case _:
            raise ValueError(f"Unsupported model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(description="UMAP visualization of Honeybee pairwise embeddings by cancer type")
    parser.add_argument('--datasets_path', type=str, default="../../data/honeybee/datasets/", help='Path to the directory containing the Honeybee dataset tensors wrt to this script')
    parser.add_argument('--model_type', type=str, choices=['repercent', 'gmlp', 'gru'], default='repercent', help='Model type to visualize')
    parser.add_argument('--wsi_embedding_mode', type=str, choices=['slide', 'patch'], default='slide', help='WSI embedding mode used by the dataset split')
    parser.add_argument('--split_seed', type=int, default=42, help='Seed of the precomputed dataset split to load')
    parser.add_argument('--select_seed', type=int, default=0, help='Checkpoint seed index to visualize (0-based)')
    parser.add_argument('--filter_cancer_types', nargs='+', default=DEFAULT_FILTER_CANCER_TYPES, help='Optional cancer types to keep, e.g. --filter_cancer_types TCGA-BRCA TCGA-LUAD or TCGA-BRCA,TCGA-LUAD. Should match training.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for embedding extraction')
    args = parser.parse_args()
    filter_cancer_types = _parse_filter_cancer_types(args.filter_cancer_types)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))

    data_config_path = os.path.join(script_dir, "../..", "configs", "data", "honeybee_data.yaml")
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)

    model_config_path = os.path.join(script_dir, "../..", "configs", "model", f"{args.model_type}_honeybee.yaml")
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    analysis_config_path = os.path.join(script_dir, "../..", "configs", "posthoc_analysis", "honeybee.yaml")
    with open(analysis_config_path, 'r') as f:
        analysis_config = yaml.safe_load(f)

    checkpoints = analysis_config[args.model_type]['checkpoints']
    n_seeds = analysis_config['hyperparameters']['n_seeds']
    assert n_seeds == len(checkpoints), (
        f"Number of seeds in hyperparameters ({n_seeds}) does not match checkpoints ({len(checkpoints)})."
    )
    assert 0 <= args.select_seed < n_seeds, f"select_seed should be between 0 and {n_seeds - 1}, but got {args.select_seed}"

    dataset_split = torch.load(
        os.path.join(script_dir, args.datasets_path, f"dataset_01_{args.wsi_embedding_mode}_split_{args.split_seed}.pt"),
        weights_only=False,
    )
    test_dataset = _filter_dataset_by_cancer_types(dataset_split['test'], filter_cancer_types)
    if filter_cancer_types is not None:
        if len(test_dataset) == 0:
            raise ValueError(f"No test samples found for cancer types: {filter_cancer_types}.")
        print(f"Filtered cancer types {filter_cancer_types}: {len(test_dataset)} test samples")
    selected_cancer_types = _available_cancer_types(test_dataset, filter_cancer_types)
    if not selected_cancer_types:
        raise ValueError(f"No cancer types were found after filtering: {filter_cancer_types}")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    checkpoint_path = os.path.join(project_root, checkpoints[args.select_seed])

    set_seed(args.select_seed + 2)
    model = _build_model(args.model_type, model_config, data_config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    modality_aliases = _build_modality_aliases(HONEYBEE_MODALITIES)

    for cancer_type in selected_cancer_types:
        pair_embeddings = _collect_pair_embeddings(
            test_loader,
            model,
            device,
            cancer_type=cancer_type,
            modality_order=HONEYBEE_MODALITIES,
        )

        available_count = 0
        for components in pair_embeddings.values():
            available_count = max(available_count, components['unique_forward'].shape[0])
        if available_count == 0:
            raise ValueError(f"Cancer type {cancer_type} was not found in the filtered test split.")

        fig_dir = os.path.join(script_dir, "figures", "embeddings", _sanitize_name(cancer_type))
        os.makedirs(fig_dir, exist_ok=True)

        for i in range(len(HONEYBEE_MODALITIES)):
            for j in range(i + 1, len(HONEYBEE_MODALITIES)):
                modality_a = HONEYBEE_MODALITIES[i]
                modality_b = HONEYBEE_MODALITIES[j]
                fig_path = os.path.join(
                    fig_dir,
                    f"umap_{args.model_type}_seed{args.select_seed}_{_sanitize_name(cancer_type)}_{modality_a}_vs_{modality_b}.pdf",
                )
                _plot_pair_umap(
                    pair_embeddings[(i, j)],
                    modality_a=modality_a,
                    modality_b=modality_b,
                    cancer_type=cancer_type,
                    random_state=args.select_seed,
                    fig_path=fig_path,
                    modality_aliases=modality_aliases,
                )
                print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
