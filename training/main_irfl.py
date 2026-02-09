import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from typing import Literal, List
from torch.utils.data import DataLoader
import torch.functional as F
from src.models.perceiver import Perceiver
from src.models.repercent import DisenEncoder, RePercENT, DisenLoss
from training.train_repercent import make_dataloaders, make_model, split_dataset
from training.train_irfl import train
from src.utils.irfl_dataset import make_dataset
from training.log_data import log_model_details, log_model_checkpoint
import math
import numpy as np
import yaml
import argparse
import time
from src.utils.helpers import set_seed

import wandb
import random




def aggregate_and_log(all_final_metrics: list):
    """
    all_final_metrics: list of dicts with keys: seed_idx, metrics (dict)
    """
    # union of metric keys
    metric_keys = sorted({k for r in all_final_metrics for k in r["metrics"].keys()})

    columns = ["seed_idx"] + metric_keys
    table = wandb.Table(columns=columns)

    # fill table
    for r in all_final_metrics:
        row = [r["seed_idx"]]
        row += [float(r["metrics"].get(k, np.nan)) for k in metric_keys]
        table.add_data(*row)

    wandb.log({"final_metrics/all_runs": table})

    # mean/std summary
    summary = {}
    for k in metric_keys:
        vals = np.array([r["metrics"].get(k, np.nan) for r in all_final_metrics], dtype=np.float32)
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            summary[f"final/{k}_mean"] = float(vals.mean())
            summary[f"final/{k}_std"] = float(vals.std())
    wandb.log(summary)


def main():
    parser = argparse.ArgumentParser(description="Train RePercENT model on the IRFL dataset")
    parser.add_argument('--datasets_path', type=str, default="../data/irfl/datasets/", help='Path to the directory containing the IRFL dataset tensors wrt to this script')
    parser.add_argument('--model_type', type=str, choices=['repercent'], default='repercent', help='Type of model to train, for now only repercent is implemented')

    # Define number of splits and seeds
    parser.add_argument('--n_seeds', type=int, default= 5, help='Number of seeds per split for model initialization and training')
    parser.add_argument('--base_seed', type=int, default= 2, help='Base seed for reproducibility')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    M = 3 # number of modalities, for the IRFL: M = 2 -> images + captions, M = 3 -> images + captions + definitions

    # Loading configurations for data, model, and training
    print("Loading configurations...")
    data_config_path = os.path.join(script_dir, "..", "configs", "data", f"irfl_data_{M}m.yaml")
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)

    model_config_path = os.path.join(script_dir, "..", "configs", "model", f"{args.model_type}_irfl_{M}m.yaml")
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    training_config_path = os.path.join(script_dir, "..", "configs", "training", f"train_irfl_{M}m.yaml")
    with open(training_config_path, 'r') as f:
        training_config = yaml.safe_load(f)

    # Load the *full dataset once*
    print("Loading datasets...")
    
    total_train_data = torch.load(os.path.join(script_dir, args.datasets_path, 'IRFL_train_tensors_2.pt'), map_location="cpu")
    total_test_data = torch.load(os.path.join(script_dir, args.datasets_path, 'IRFL_test_tensors_2.pt'), map_location="cpu")

    total_train_data_aug = torch.load(os.path.join(script_dir, args.datasets_path, 'IRFL_train_tensors_aug_2.pt'), map_location="cpu")
    total_test_data_aug = torch.load(os.path.join(script_dir, args.datasets_path, 'IRFL_test_tensors_aug_2.pt'), map_location="cpu")
    train_dataset, train_data_dict = make_dataset(total_data= total_train_data | total_train_data_aug, num_modalities= data_config["create_data"]["M"], data_type='train', include_original=True)
    test_dataset, test_data_dict = make_dataset(total_data= total_test_data | total_test_data_aug, num_modalities= data_config["create_data"]["M"], data_type='test', include_original=True)


    group_name = time.strftime("%Y-%m-%d_%H-%M-%S") + f"_IRFL_{args.model_type}_seeds_{args.n_seeds}"
    # Initialize list to store final metrics across all runs
    all_final_metrics = []
    print(f"Starting training runs...n_seeds: {args.n_seeds}, base_seed: {args.base_seed}")
    # Outer loop over different seeds
    for seed_idx in range(args.n_seeds):
        train_seed = args.base_seed + seed_idx
        set_seed(train_seed)

        g = torch.Generator()
        g.manual_seed(seed_idx)

        
        # split the train dataset into train and validation sets
        temp_train_dataset, temp_val_dataset = split_dataset(train_dataset, test_size= 0.1, generator= g)
        # dataloaders
        train_loader, val_loader = make_dataloaders(temp_train_dataset, temp_val_dataset, batch_size=training_config["training"]["batch_size"], generator=g)
        test_loader = DataLoader(test_dataset, batch_size=training_config["training"]["batch_size"], shuffle=False)


        # Initialize wandb run and log hyperparameters
        run = wandb.init(
            project=data_config["wandb"]["project"],
            group=group_name,
            name=f"{group_name}_seed_{seed_idx}",
            config={
                "n_seeds": args.n_seeds, "base_seed": args.base_seed,
                "model_type": args.model_type,
            }
        )
        print("INIT RUN:", wandb.run.id, wandb.run.name)
        print("EPOCH RUN:", wandb.run.id, wandb.run.name)

        log_model_details(run, model_name=args.model_type, data_config=data_config_path, model_config=model_config_path, training_config=training_config_path)

        # model creation
        disenEncoders = [make_model(model_config, data_config, modality=m + 1, M=data_config["create_data"]["M"]) for m in range(data_config["create_data"]["M"])]
        model = RePercENT(M=data_config["create_data"]["M"],
                        disenEncoder=disenEncoders,
                        recon= training_config["disen_loss"]["recon"],
                        disen_mapping=model_config["repercent"]["disen_mapping"]).to(device)

        disen_loss = DisenLoss(alpha=training_config["disen_loss"]["alpha"],
                            lmd=training_config["disen_loss"]["lmd"],
                            lmd_end_value=training_config["disen_loss"]["lmd_end_value"],
                            M=data_config["create_data"]["M"],
                            recon= training_config["disen_loss"]["recon"])

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_config["optimizer"]["lr"],
            weight_decay=training_config["optimizer"]["weight_decay"]
        )

        # run key for logging
        run_key = f"seed{seed_idx}"

        # Logging identifiers
        wandb.log({
            "meta/seed_idx": seed_idx
        })

        # TRAIN
        final_metrics = train(train_loader, val_loader, test_loader, model, \
                            optimizer, disen_loss, training_config["training"]["n_epochs"], \
                            device, checkpoint_dir=os.path.join(script_dir, '..', 'checkpoints', 'irfl', run.name, run_key))

        # Store + log final snapshot table
        all_final_metrics.append({
            "seed_idx": seed_idx,
            "metrics": final_metrics,
        })
        wandb.finish()


    # global summary run
    run = wandb.init(project= data_config["wandb"]["project"], 
                    group=group_name, name= f"aggregate_{args.model_type}", 
                    reinit=True, config= {"n_seeds": args.n_seeds, "base_seed": args.base_seed, "model_type": args.model_type})
    
    aggregate_and_log(all_final_metrics)
    wandb.finish()

if __name__ == "__main__":
    main()