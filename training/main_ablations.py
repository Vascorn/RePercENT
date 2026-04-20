import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from typing import Literal, List
from torch.utils.data import DataLoader
from src.utils.synthetic_dataset import GenerateTokenizedData, MultimodalDataset, save_dataset, save_data_split, GeneratePermData, GenerateData
from src.models.perceiver import Perceiver
from src.models.repercent import DisenEncoder, RePercENT, DisenLoss
from src.utils.helpers import set_seed, extract_latents_and_labels
from training.train_repercent import split_dataset, make_dataloaders, train, make_model
from training.log_data import log_model_details, log_model_checkpoint, log_dataset
from training.train_jointopt_2m import make_model_jointopt
from training.main import split_dataset_seeded
import math
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import yaml
import argparse
import time

import wandb
import random


def _as_float(value):
    if value is None:
        return np.nan
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return np.nan
        value = value.detach().cpu().item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def aggregate_and_log(all_metrics: list, table_name: str):
    """
    Aggregate final metrics across data splits and log the result as a W&B table.

    all_metrics: list of dicts with keys:
      split_idx, split_seed, train_seed, metrics (dict)
    """
    if not all_metrics:
        print("No metrics were collected; skipping aggregate wandb table.")
        return

    metric_keys = sorted({k for r in all_metrics for k in (r.get("metrics") or {}).keys()})
    split_columns = [f"split_{r['split_idx']}" for r in all_metrics]
    columns = ["metric", "mean", "std", "n_splits"] + split_columns
    table = wandb.Table(columns=columns)

    summary = {}
    for metric_key in metric_keys:
        split_values = np.array(
            [_as_float((r.get("metrics") or {}).get(metric_key, np.nan)) for r in all_metrics],
            dtype=np.float64,
        )
        finite_values = split_values[np.isfinite(split_values)]
        if len(finite_values) == 0:
            continue

        mean_value = float(finite_values.mean())
        std_value = float(finite_values.std())
        row_values = [None if not np.isfinite(v) else float(v) for v in split_values]
        table.add_data(metric_key, mean_value, std_value, int(len(finite_values)), *row_values)

        summary[f"final/{metric_key}_mean"] = mean_value
        summary[f"final/{metric_key}_std"] = std_value

    wandb.log({table_name: table})
    if summary:
        wandb.log(summary)


def main():
    parser = argparse.ArgumentParser(description="Train RePercENT or Jointopt model on synthetic data")
    parser.add_argument('--model_type', type=str, choices=['jointopt', 'repercent'], default='repercent', help='Type of model to train')

    # Define number of splits and seeds
    parser.add_argument('--k1', type=int, default= 3, help='Number of different train/test splits')
    parser.add_argument('--base_seed', type=int, default=2, help='Base seed for reproducibility')
    parser.add_argument('--M', type=int, default=5, help='Number of modalities in the dataset and model')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    

    M = args.M # number of modalities, used for model creation and dataset generation
    # Loading configurations for data, model, and training
    data_config_path = os.path.join(script_dir, "..", "configs", "data", f"synthetic_data_{M}m.yaml")
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)

    model_config_path = os.path.join(script_dir, "..", "configs", "model", f"{args.model_type}_{M}m.yaml")
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    training_config_path = os.path.join(script_dir, "..", "configs", "training", f"train_synthetic_{M}m.yaml")
    with open(training_config_path, 'r') as f:
        training_config = yaml.safe_load(f)

    

    # Load the dataset
    load_path = os.path.join(script_dir, "..", "data", "repercent_synthetic", f"dataset2{M}")
    dataset = torch.load(os.path.join(load_path, "dataset.pt"), weights_only=False)
    
    
    group_name = f"{args.model_type}_splits_{args.k1}" if args.model_type == "repercent" else f"_{model_config['shared_encoder']['type']}_splits_{args.k1}"
    group_name += f"_{M}M" # modality number identifier for wandb grouping
    


    # Ablate over Semantic positional encodings (SE) and Group Slot attention (GSA)
    for (add_se, add_gsa) in [(True, False), (False, True), (False, False)]:
        # Update model config for current ablation
        model_config["perceiver"]["use_slot_attn"] = add_gsa
        add_se_str = "w_SE" if add_se else "wo_SE"
        add_gsa_str = "w_GSA" if add_gsa else "wo_GSA"
        print(f"Running ablation with {add_se_str} and {add_gsa_str}")
        run_suffix = f"{add_se_str}_{add_gsa_str}"


        # Data splits
        # Initialize list to store final metrics across all runs
        all_final_metrics = []
        for split_idx in range(args.k1):
            split_seed = args.base_seed + 10_000 + split_idx
            
            # deterministic split
            train_dataset, test_dataset, val_dataset = split_dataset_seeded(dataset, \
                                                                            test_size=training_config["training"]["test_size"], \
                                                                            val_size= training_config["training"]["val_size"], \
                                                                            seed=split_seed)


            train_seed = args.base_seed + 100 * split_idx
            set_seed(train_seed)
            generator = torch.Generator().manual_seed(train_seed)
            # dataloaders
            train_loader, test_loader, val_loader = make_dataloaders(train_dataset, test_dataset, val_dataset= val_dataset, batch_size=training_config["training"]["batch_size"], generator=generator)

            # Initialize wandb run and log hyperparameters
            run = wandb.init(
                project= "repercent_ablation_synthetic",
                group=group_name,
                name=f"{group_name}_split_{split_idx}_{run_suffix}",
                config={
                    "k1": args.k1, "base_seed": args.base_seed,
                    "model_type": args.model_type,
                    "add_se": add_se, "add_gsa": add_gsa
                }
            )

            log_model_details(run, model_name=args.model_type, data_config=data_config_path, model_config=model_config_path, training_config=training_config_path)

            # model creation based on model_type
            if args.model_type == 'jointopt':
                model = make_model_jointopt(model_config).to(device)
            else:
                disenEncoders = [make_model(model_config, data_config, modality=m + 1, M=data_config["create_data"]["M"]) for m in range(data_config["create_data"]["M"])]
                model = RePercENT(M=data_config["create_data"]["M"],
                                disenEncoder=disenEncoders,
                                disen_mapping=model_config["repercent"]["disen_mapping"],
                                add_pos_encoding= add_se).to(device)

            disen_loss = DisenLoss(alpha=training_config["disen_loss"]["alpha"],
                                    beta=training_config["disen_loss"]["beta"],
                                    lmd=training_config["disen_loss"]["lmd"],
                                    lmd_start_value=training_config["disen_loss"]["lmd_start_value"],
                                    lmd_end_value=training_config["disen_loss"]["lmd_end_value"],
                                    lmd_n_iterations=training_config["disen_loss"]["lmd_n_iterations"],
                                    lmd_start_iteration=training_config["disen_loss"]["lmd_start_iteration"],
                                    M=data_config["create_data"]["M"])

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=training_config["optimizer"]["lr"],
                weight_decay=training_config["optimizer"]["weight_decay"]
            )
            
            # run key for logging
            run_key = f"split{split_idx}"

            # Logging identifiers
            wandb.log({
                "meta/split_idx": split_idx,
                "meta/split_seed": split_seed,
                "meta/train_seed": train_seed,
            })

            # TRAIN
            final_metrics = train(train_loader, test_loader, model, optimizer, disen_loss, \
                                training_config["training"]["n_epochs"], \
                                device, val_loader= val_loader, \
                                checkpoint_dir=os.path.join(script_dir, '..', 'checkpoints', 'repercent_synthetic', run.name, run_key), \
                                generator= generator)

            # Store + log final snapshot table
            all_final_metrics.append({
                "split_idx": split_idx,
                "split_seed": split_seed,
                "train_seed": train_seed,
                "metrics": final_metrics,
            })
            wandb.finish()
        
        # After all splits are done, log the aggregated final metrics across splits as a summary table in wandb
        aggregate_table_name = f"{group_name}_{run_suffix}"
        run = wandb.init(
            project="repercent_ablation_synthetic",
            group=group_name,
            name=f"aggregate_{aggregate_table_name}",
            reinit=True,
            config={
                "k1": args.k1, "base_seed": args.base_seed,
                "model_type": args.model_type,
                "add_se": add_se, "add_gsa": add_gsa,
                "table_name": aggregate_table_name,
            }
        )
        aggregate_and_log(all_final_metrics, table_name=aggregate_table_name)
        wandb.finish()
        
    

if __name__ == "__main__":
    main()
