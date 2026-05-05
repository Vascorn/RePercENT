import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.functional as F
import typing
from typing import Literal, List
from src.models import repercent, jointopt
from src.models.repercent import RePercENT
from src.utils.helpers import ProbeEvaluator, extract_latents_and_labels, linear_probe, non_linear_probe, regression_probe, plot_confusion_matrix, plot_pairwise_confusion_matrices
from training.train_repercent import make_model
from training.train_jointopt_2m import make_model_jointopt
from posthoc.synthetic.helper_metrics import linear_probe_disentanglement_metric
import yaml
import argparse
import wandb
from src.utils.helpers import set_seed
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from training.main import aggregate_and_log
from torch.profiler import profile, ProfilerActivity
from itertools import combinations
from posthoc.plotting_config import paper_plot_context


def calculate_flops(model, loader, device):
    model.eval()

    # Grab one batch and calculate FLOPs using PyTorch profiler
    X_batch, _, _, _, _ = next(iter(loader))
    X_batch = [x.to(device) for x in X_batch]

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        with_flops=True,
        profile_memory=False,
    ) as prof:
        with torch.no_grad():
            model(X_batch, mask = [None for _ in range(len(X_batch))])
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Aggregate FLOPs over all ops
    total_flops = sum(evt.flops for evt in prof.key_averages() if evt.flops is not None)
    print(f"Estimated FLOPs: {total_flops / 1e6:.2f} MFLOPs")
    print(prof.key_averages().table(sort_by="flops", row_limit=15))

    return total_flops


def _latex_label(key):
    kind, indices = key.split("_", maxsplit=1)
    indices = indices.replace("_", "")
    return rf"$y_{{{kind}_{{{indices}}}}}$"

def _latex_component(key):
    kind, indices = key.split("_", maxsplit=1)
    indices = indices.replace("_", "")
    return rf"${kind}_{{{indices}}}$"


def _build_linear_results_table(all_linear_results, components):
    columns = ["split_idx", "seed_idx", "split_seed", "train_seed", "metric", "label", "component", "value"]
    table = wandb.Table(columns=columns)
    for run_results in all_linear_results:
        for metric_name, metric_results in run_results["linear_results"].items():
            for label_key, values in metric_results.items():
                values = np.asarray(values, dtype=float)
                for comp_idx, component_key in enumerate(components):
                    if comp_idx < len(values):
                        table.add_data(
                            run_results["split_idx"],
                            run_results["seed_idx"],
                            run_results["split_seed"],
                            run_results["train_seed"],
                            metric_name,
                            label_key,
                            component_key,
                            float(values[comp_idx]),
                        )
    return table


def _aggregate_linear_probe_acc(all_linear_results, components):
    mean_acc = {}
    std_acc = {}
    for label_key in components:
        values = [
            np.asarray(run_results["linear_results"]["acc"][label_key], dtype=float)
            for run_results in all_linear_results
            if label_key in run_results["linear_results"]["acc"]
        ]
        if not values:
            continue
        stacked = np.stack(values, axis=0)
        mean_acc[label_key] = np.nanmean(stacked, axis=0)
        std_acc[label_key] = np.nanstd(stacked, axis=0)
    return mean_acc, std_acc


def _summary_cell_text_color(value, cmap_name="PuBu", vmin=50.0, vmax=100.0):
    cmap = plt.get_cmap(cmap_name)
    normalized = np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0)
    r, g, b, _ = cmap(normalized)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.48 else "#1a1a1a"


def _resolve_output_dir(path, script_dir):
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(script_dir, path))


def _plot_summary_pairwise_confusion_matrices(mean_acc, std_acc, M, components):
    comp_idx = {key: idx for idx, key in enumerate(components)}
    pairs = list(combinations(range(M), 2))
    x_shape = M if M % 2 else M // 2
    y_shape = M - 1 if (M - 1) % 2 else M // 2
    x_shape, y_shape = (y_shape, x_shape) if x_shape > y_shape else (x_shape, y_shape)

    with paper_plot_context():
        fig, axes = plt.subplots(
            x_shape,
            y_shape,
            figsize=(3.35 * y_shape + 0.5, 3.55 * x_shape),
            constrained_layout=True,
        )
        fig.patch.set_facecolor("white")
        axes = np.atleast_1d(axes).ravel()

        for pair_id, (i, j) in enumerate(pairs):
            col_keys = [f"u_{i+1}{j+1}", f"u_{j+1}{i+1}", f"s_{i+1}{j+1}"]
            row_keys = col_keys
            mean_mat = np.full((len(row_keys), len(col_keys)), np.nan, dtype=float)
            std_mat = np.full_like(mean_mat, np.nan)

            for r_idx, row_key in enumerate(row_keys):
                if row_key not in mean_acc:
                    continue
                for c_idx, col_key in enumerate(col_keys):
                    col_index = comp_idx.get(col_key)
                    if col_index is None or col_index >= mean_acc[row_key].shape[0]:
                        continue
                    mean_mat[r_idx, c_idx] = mean_acc[row_key][col_index]
                    std_mat[r_idx, c_idx] = std_acc[row_key][col_index]

            ax = axes[pair_id]
            display_cols = [_latex_component(key) for key in col_keys]
            display_rows = [_latex_label(key) for key in row_keys]
            sns.heatmap(
                mean_mat,
                annot=False,
                cmap="PuBu",
                xticklabels=display_cols,
                yticklabels=display_rows,
                cbar=False,
                vmin=50,
                vmax=100,
                square=True,
                linewidths=0.8,
                linecolor="white",
                mask=~np.isfinite(mean_mat),
                ax=ax,
            )

            for r_idx in range(mean_mat.shape[0]):
                for c_idx in range(mean_mat.shape[1]):
                    if not np.isfinite(mean_mat[r_idx, c_idx]):
                        continue
                    text_color = _summary_cell_text_color(mean_mat[r_idx, c_idx])
                    ax.text(
                        c_idx + 0.5,
                        r_idx + 0.41,
                        f"{mean_mat[r_idx, c_idx]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=11,
                        fontweight="semibold",
                        color=text_color,
                    )
                    ax.text(
                        c_idx + 0.5,
                        r_idx + 0.69,
                        rf"$\pm$ {std_mat[r_idx, c_idx]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8.5,
                        color=text_color,
                    )

            ax.set_title(rf"$X_{{{i+1}}} \leftrightarrow X_{{{j+1}}}$", pad=8)
            ax.set_xlabel("Components", labelpad=5)
            ax.set_ylabel("Labels", labelpad=5)
            ax.tick_params(axis="both", length=0, pad=2)
            plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
            plt.setp(ax.get_yticklabels(), rotation=0, va="center")
            for spine in ax.spines.values():
                spine.set_visible(False)

        for ax in axes[len(pairs):]:
            ax.axis("off")

        norm = plt.Normalize(vmin=50, vmax=100)
        sm = plt.cm.ScalarMappable(cmap="PuBu", norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[:len(pairs)], shrink=0.72, pad=0.015)
        cbar.set_label("Linear probe accuracy (%)", labelpad=8, fontsize=12)
        cbar.ax.tick_params(labelsize=12, length=3)

    return fig


def main():
    
    parser = argparse.ArgumentParser(description="Post hoc evaluation of trained models on synthetic data")
    parser.add_argument('--datasets_path', type=str, default="../../data/irfl/datasets/", help='Path to the directory containing the IRFL dataset tensors wrt to this script')
    parser.add_argument('--model_type', type=str, choices= ['repercent', 'jointopt'], default='repercent', help='Type of model to train, for now only repercent is implemented')
    parser.add_argument('--enc_type', type=str, choices= ['gMLP', 'MLP', 'GRU'], default= 'MLP', help= "The different baseline encoders, if <model_type> is jointopt. This argument \
                                                                                                        is inactive, if model type is repercent")
    parser.add_argument('--M', type=int, default= 2, help='Number of modalities in the synthetic setup.')
    
    # Define number of splits and seeds - these should exactly match the training ones for reproducibility
    parser.add_argument('--k1', type=int, default= 3, help='Number of different train/test splits')
    parser.add_argument('--k2', type=int, default= 2, help='Number of training seeds per split')
    parser.add_argument('--base_seed', type=int, default=2, help='Base seed for reproducibility')
    parser.add_argument('--save_figures_path', type=str, default='./figures/confusion_matrices/', help='Path to save confusion matrix figures')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_figures_path = _resolve_output_dir(args.save_figures_path, script_dir)
    
    M = args.M # number of modalities
    
    # Loading configurations for data, model, and training
    print("Loading configurations...")
    data_config_path = os.path.join(script_dir, "../..", "configs", "data", f"synthetic_data_{M}m.yaml")
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)

    model_config_path = os.path.join(script_dir, "../..", "configs", "model", f"{args.model_type}_{M}m.yaml")
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    training_config_path = os.path.join(script_dir, "../..", "configs", "training", f"train_synthetic_{M}m.yaml")
    with open(training_config_path, 'r') as f:
        training_config = yaml.safe_load(f)

    analysis_config_path = os.path.join(script_dir, "../..", "configs", "posthoc_analysis", f"synthetic_{M}m.yaml")
    with open(analysis_config_path, 'r') as f:
        analysis_config = yaml.safe_load(f)


    if args.model_type == "jointopt" and (args.enc_type.lower() != model_config["shared_encoder"]["type"].lower()):
        raise ValueError(f"Encoder type argument does not match the encoder type specified in the model config. Please check your arguments and configs. \
                        Argument encoder type: {args.enc_type}, Model config encoder type: {model_config['shared_encoder']['type']}")

    data_path = os.path.join(script_dir, "../..", "data", "repercent_synthetic", f"dataset2{M}") # change accordingly
    all_metrics_summary = []
    all_linear_results = []
    components = None
    for split_idx in range(args.k1):
        # load data-split
        split_path = os.path.join(data_path, f"data_split_{split_idx}.pt")
        dataset_split = torch.load(split_path, weights_only= False)
        
        train_dataset, test_dataset = dataset_split["train_dataset"], dataset_split["test_dataset"]

        for seed_idx in range(args.k2):
            # set seed for reproducibility
            seed = args.base_seed + split_idx * args.k2 + seed_idx
            set_seed(seed)
            generator = torch.Generator().manual_seed(seed)

            train_loader = DataLoader(train_dataset, batch_size= training_config["training"]["batch_size"], generator=generator)
            test_loader = DataLoader(test_dataset, batch_size= training_config["training"]["batch_size"], generator=generator)
            
            
    
            # model creation based on model_type
            if args.model_type == 'jointopt':
                model = make_model_jointopt(model_config).to(device)
                # load model state dictionary
                model_state_dict = analysis_config[args.model_type][args.enc_type.lower()]["checkpoints"][split_idx * args.k2 + seed_idx]
            else:
                disenEncoders = [make_model(model_config, data_config, modality=m + 1, M=data_config["create_data"]["M"]) for m in range(data_config["create_data"]["M"])]
                model = RePercENT(M=data_config["create_data"]["M"],
                                disenEncoder=disenEncoders,
                                disen_mapping=model_config["repercent"]["disen_mapping"]).to(device)

                model_state_dict = analysis_config[args.model_type]["checkpoints"][split_idx * args.k2 + seed_idx]

            
            temp_state_dict = torch.load(os.path.join(script_dir, "../..", model_state_dict), map_location=device)
            model.load_state_dict(temp_state_dict['model_state_dict'])


            train_data_dict = extract_latents_and_labels(model, train_loader, device)
            test_data_dict = extract_latents_and_labels(model, test_loader, device)

            if components is None:
                components = list(train_data_dict['Labels_U'].keys()) + list(train_data_dict['Labels_S'].keys())

            evaluator = ProbeEvaluator(linear_probe= linear_probe, regression_probe= regression_probe)
            evaluator.set_data(train_data_dict= train_data_dict, val_data_dict= test_data_dict, M= M)
            
            linear_results = evaluator.calculate_linear_probe()
            reg_results = evaluator.calculate_reg_probe()

            metrics_summary = evaluator.mean_metrics(linear_results, reg_results, M= M)

            all_metrics_summary.append({
                "split_idx": split_idx,
                "seed_idx": seed_idx,
                "split_seed": split_idx,
                "train_seed": seed,
                "metrics": metrics_summary
            })
            all_linear_results.append({
                "split_idx": split_idx,
                "seed_idx": seed_idx,
                "split_seed": split_idx,
                "train_seed": seed,
                "linear_results": {
                    metric_name: {
                        label_key: np.asarray(values, dtype=float).copy()
                        for label_key, values in metric_results.items()
                    }
                    for metric_name, metric_results in linear_results.items()
                },
            })
            
            
            print(f"Evaluation of split: {split_idx} and seed: {seed_idx} complete!")
    

    run = wandb.init(project= "posthoc_" + model_config["wandb"]["project"], 
                    name= f"aggregate_{args.model_type}" if args.model_type == "repercent" else f"aggregate_{model_config['shared_encoder']['type']}", 
                    reinit=True, config={"k1": args.k1, "k2": args.k2, "base_seed": args.base_seed, "model_type": args.model_type, "enc_type": args.enc_type})

    aggregate_and_log(all_metrics_summary) 
    if all_linear_results and components is not None:
        wandb.log({"linear_probe/all_runs": _build_linear_results_table(all_linear_results, components)})
        mean_acc, std_acc = _aggregate_linear_probe_acc(all_linear_results, components)
        fig = _plot_summary_pairwise_confusion_matrices(mean_acc, std_acc, M=M, components=components)
        
        fig_path = os.path.join(save_figures_path, f"summary_pairwise_confusion_matrices_{args.model_type}_{args.enc_type}_{M}m.pdf" if args.model_type == "jointopt" else f"summary_pairwise_confusion_matrices_{args.model_type}_{M}m.pdf")
        os.makedirs(save_figures_path, exist_ok=True)
        fig.savefig(fig_path, bbox_inches='tight')
        print(f"Saved summary pairwise confusion matrices to: {fig_path}")
        wandb.log({"linear_pairwise_confusion_matrices_summary": wandb.Image(fig)})
        plt.close(fig)

    # Log model parameters and flops as well
    model_params = sum(p.numel() for p in model.parameters())
    model_flops = calculate_flops(model, test_loader, device)
    
    wandb.log({"model_params": model_params, "model_flops": model_flops})
    wandb.finish()

if __name__ == "__main__":
    main()
