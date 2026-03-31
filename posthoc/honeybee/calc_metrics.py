import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.functional as F
import typing
from typing import Literal, List
from collections import defaultdict
from src.models import repercent, jointopt
from src.models.repercent import RePercENT
from posthoc.honeybee.helper_metrics import evaluate_model_cancer_type, evaluate_model_survival_analysis
from posthoc.honeybee.helper_vis import plot_cancer_type_distribution
from training.train_repercent import make_dataloaders, make_model
from training.train_jointopt_2m import make_model_jointopt
import yaml
import argparse
import wandb
from src.utils.helpers import set_seed
import numpy as np


def _to_float(value):
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _aggregate_metrics_across_seeds(seed_reports):
    aggregated = {}

    for seed_report in seed_reports:
        for component_name, component_metrics in seed_report.items():
            component_name = str(component_name)
            component_entry = aggregated.setdefault(component_name, {"overall": [], "per_cancer_type": {}})
            component_entry["overall"].append(_to_float(component_metrics["overall"]))

            for cancer_type, score in component_metrics["per_cancer_type"].items():
                cancer_type = str(cancer_type)
                component_entry["per_cancer_type"].setdefault(cancer_type, [])
                component_entry["per_cancer_type"][cancer_type].append(_to_float(score))

    summary = {}
    for component_name, component_metrics in aggregated.items():
        summary[component_name] = {
            "overall": {
                "mean": float(np.mean(component_metrics["overall"])),
                "std": float(np.std(component_metrics["overall"], ddof=1)) if len(component_metrics["overall"]) > 1 else 0.0,
            },
            "per_cancer_type": {},
        }

        for cancer_type, scores in component_metrics["per_cancer_type"].items():
            summary[component_name]["per_cancer_type"][cancer_type] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            }

    return summary


def _build_wandb_summary_table(summary_metrics):
    cancer_types = sorted(
        {
            str(cancer_type)
            for component_metrics in summary_metrics.values()
            for cancer_type in component_metrics["per_cancer_type"].keys()
        }
    )
    columns = [str(column) for column in ["component", "overall", *cancer_types]]
    table = wandb.Table(columns=columns)

    for component_name in sorted(summary_metrics.keys()):
        component_metrics = summary_metrics[component_name]
        row = [
            component_name,
            f'{component_metrics["overall"]["mean"]:.4f} ± {component_metrics["overall"]["std"]:.4f}',
        ]

        for cancer_type in cancer_types:
            stats = component_metrics["per_cancer_type"].get(cancer_type)
            if stats is None:
                row.append("N/A")
            else:
                row.append(f'{stats["mean"]:.4f} ± {stats["std"]:.4f}')

        table.add_data(*row)

    return table


def _aggregate_survival_metrics_across_seeds(seed_reports):
    aggregated = {}

    for seed_report in seed_reports:
        for component_name, component_metrics in seed_report.items():
            component_entry = aggregated.setdefault(str(component_name), defaultdict(lambda: defaultdict(list)))

            for cancer_type, cancer_metrics in component_metrics.items():
                for metric_name, metric_value in cancer_metrics.items():
                    component_entry[str(cancer_type)][str(metric_name)].append(_to_float(metric_value))

    summary = {}
    for component_name, component_metrics in aggregated.items():
        summary[component_name] = {}
        for cancer_type, cancer_metrics in component_metrics.items():
            summary[component_name][cancer_type] = {}
            for metric_name, values in cancer_metrics.items():
                summary[component_name][cancer_type][metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                }

    return summary


def _build_survival_summary_table(summary_metrics):
    metric_names = ["c_index", "test_loss", "n_train", "n_test", "n_bins"]
    columns = ["component", "cancer_type", *metric_names]
    table = wandb.Table(columns=columns)

    for component_name in sorted(summary_metrics.keys()):
        component_metrics = summary_metrics[component_name]

        for cancer_type in sorted(component_metrics.keys()):
            cancer_row = [component_name, cancer_type]
            for metric_name in metric_names:
                stats = component_metrics[cancer_type].get(metric_name)
                if stats is None:
                    cancer_row.append("N/A")
                else:
                    cancer_row.append(f'{stats["mean"]:.4f} ± {stats["std"]:.4f}')
            table.add_data(*cancer_row)

    return table


def main():
    
    parser = argparse.ArgumentParser(description="Train RePercENT model on the IRFL dataset")
    parser.add_argument('--datasets_path', type=str, default="../../data/honeybee/datasets/", help='Path to the directory containing the IRFL dataset tensors wrt to this script')
    parser.add_argument('--model_type', type=str, choices=['repercent', 'gmlp', 'gru'], default='repercent', help='Type of model to train, for now only repercent is implemented')
    parser.add_argument('--wsi_embedding_mode', type=str, choices=['slide', 'patch'], default='slide', help='Method for aggregating WSI embeddings, either path level or slide level')
    parser.add_argument('--split_seed', type=int, default= 42, help='Seed for reproducible dataset splits, should match the seed used during training for loading the correct split')
    # Define number of splits and seeds
    parser.add_argument('--base_seed', type=int, default= 2, help='Base seed for reproducibility')
    parser.add_argument('--cancer_classification', type=bool, default=False, help='Whether to perform cancer type classification evaluation')
    parser.add_argument('--survival_analysis', type=bool, default=True, help='Whether to perform survival analysis evaluation')
    parser.add_argument('--survival_cancer_types', type=str, nargs='+', default=None, help='Specific cancer types for survival analysis, e.g. TCGA-BRCA TCGA-LUAD')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Loading configurations for data, model, and training
    print("Loading configurations...")
    data_config_path = os.path.join(script_dir, "../..", "configs", "data", f"honeybee_data.yaml")
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)

    model_config_path = os.path.join(script_dir, "../..", "configs", "model", f"{args.model_type}_honeybee.yaml")
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    analysis_config_path = os.path.join(script_dir, "../..", "configs", "posthoc_analysis", f"honeybee.yaml")
    with open(analysis_config_path, 'r') as f:
        analysis_config = yaml.safe_load(f)
    

    # Load the full dataset once and create a fixed stratified split by cancer type.
    dataset_split = torch.load(os.path.join(script_dir, args.datasets_path, f"dataset_01_{args.wsi_embedding_mode}_split_{args.split_seed}.pt"), weights_only=False)
    train_dataset = dataset_split['train']
    test_dataset = dataset_split['test']

    # Create loaders
    test_loader = DataLoader(test_dataset, batch_size= 32, shuffle=False, generator= torch.Generator().manual_seed(args.base_seed))
    train_loader = DataLoader(train_dataset, batch_size= 32, shuffle=True, generator= torch.Generator().manual_seed(args.base_seed))    

    plot_cancer_type_distribution(test_loader = test_loader, 
                                train_loader = train_loader, 
                                script_dir = script_dir,
                                train_color = "tab:blue",
                                test_color = "tab:red")
    
    # seed check
    n_seeds = analysis_config['hyperparameters']['n_seeds']
    assert n_seeds == len(analysis_config[args.model_type]['checkpoints']), f"Number of seeds in hyperparameters ({n_seeds}) does not match number of checkpoints specified for {args.model_type} ({len(analysis_config[args.model_type]['checkpoints'])})"


    # define project root for loading checkpoints
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # init device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # init results storage
    complete_report = {f"seed_{i}": {} for i in range(args.base_seed, args.base_seed + n_seeds)}
    seed_reports = []
    survival_seed_reports = []
    for seed_idx, checkpoint_path in enumerate(analysis_config[args.model_type]['checkpoints']):
        temp_seed = args.base_seed + seed_idx

        print(f"Evaluating seed {seed_idx}...")
        set_seed(temp_seed)

        # Initialize model and load weights
        match args.model_type:
            case "repercent":
                disenEncoders = [make_model(model_config, data_config, modality=m + 1, M=data_config["create_data"]["M"]) for m in range(data_config["create_data"]["M"])]
                model = RePercENT(M=data_config["create_data"]["M"],
                                disenEncoder= disenEncoders,
                                disen_mapping= model_config["repercent"]["disen_mapping"]).to(device)
            case "gmlp":
                model = make_model_jointopt(model_config).to(device)
            case "gru":
                model = make_model_jointopt(model_config).to(device)
            case _:
                raise ValueError(f"Unsupported model type: {args.model_type}")
        
        temp_state_dict = torch.load(os.path.join(project_root, checkpoint_path), map_location=device)
        model.load_state_dict(temp_state_dict['model_state_dict'])

        model.to(device)
        
        if args.cancer_classification:
             temp_metrics = evaluate_model_cancer_type(test_loader, model, device)
             complete_report[f"seed_{temp_seed}"] = temp_metrics
             seed_reports.append(temp_metrics)
        if args.survival_analysis:
           
            survival_metrics = evaluate_model_survival_analysis(
                train_loader,
                test_loader,
                model,
                device,
                cancer_types=args.survival_cancer_types,
            )
            complete_report.setdefault("survival_analysis", {})[f"seed_{temp_seed}"] = survival_metrics

            survival_seed_reports.append(survival_metrics)

    wandb.init(
        project=analysis_config["wandb"]["project"],
        name=f"{args.model_type}_cancer_type_probe_summary",
        config={
            "model_type": args.model_type,
            "split_seed": args.split_seed,
            "base_seed": args.base_seed,
            "n_seeds": n_seeds,
            "wsi_embedding_mode": args.wsi_embedding_mode,
        },
    )

    if args.cancer_classification:
        complete_report["summary"] = _aggregate_metrics_across_seeds(seed_reports)
        summary_table = _build_wandb_summary_table(complete_report["summary"])
        wandb.log({"cancer_type_component_summary": summary_table})
    if args.survival_analysis:
        complete_report["survival_summary"] = _aggregate_survival_metrics_across_seeds(survival_seed_reports)
        survival_summary_table = _build_survival_summary_table(complete_report["survival_summary"])

        wandb.log({"survival_analysis_component_summary": survival_summary_table})
    wandb.finish()

if __name__ == "__main__":
    main()
