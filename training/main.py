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


def create_dataset_synth(data_config: dict= None)-> MultimodalDataset:
    '''
    Create synthetic dataset based on the data configuration and save it to the specified path.
    Args:
        data_config: Configuration dictionary for the data.
    '''
    gen_data = GenerateTokenizedData(N_data= data_config["create_data"]["N_data"], trans_type= data_config["create_data"]["trans_type"], latent_dim= data_config["create_data"]["latent_dim"], M = data_config["create_data"]["M"])
    gen_data.create_dataset(dist= data_config["create_data"]["dist"], ts= data_config["create_data"]["ts"], gammas= data_config["create_data"]["gammas"], normalize= data_config["create_data"]["normalize"], sigma= data_config["create_data"]["sigma"])
    dataset = MultimodalDataset(total_data= gen_data.dataset_dict['total_data'], labels_u= gen_data.dataset_dict['labels_u'], labels_s= gen_data.dataset_dict['labels_s'], t_u= gen_data.dataset_dict['t_u'], t_s = gen_data.dataset_dict['t_s'])

    return dataset



def split_dataset_seeded(dataset, test_size: float, val_size: float, seed: int):
    n_total = len(dataset)
    n_test = int(round(n_total * test_size))
    n_val = int(round(n_total * val_size))
    n_train = n_total - n_test - n_val
    g = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_test, n_val], generator=g)


def aggregate_and_log(all_final_metrics: list):
    """
    all_final_metrics: list of dicts with keys:
      split_idx, seed_idx, split_seed, train_seed, metrics (dict)
    """
    # union of metric keys
    metric_keys = sorted({k for r in all_final_metrics for k in r["metrics"].keys()})

    columns = ["split_idx", "seed_idx", "split_seed", "train_seed"] + metric_keys
    table = wandb.Table(columns=columns)

    # fill table
    for r in all_final_metrics:
        row = [r["split_idx"], r["seed_idx"], r["split_seed"], r["train_seed"]]
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
    parser = argparse.ArgumentParser(description="Train RePercENT or Jointopt model on synthetic data")
    parser.add_argument('--save_data', type=bool, default=False, help='Whether to save the generated synthetic dataset')
    parser.add_argument('--save_data_split', type=bool, default=False)
    parser.add_argument('--load_data', type=bool, default=True)
    parser.add_argument('--log_dataset_artifact', type=bool, default=False)
    parser.add_argument('--model_type', type=str, choices=['jointopt', 'repercent'], default='repercent', help='Type of model to train')

    # Define number of splits and seeds
    parser.add_argument('--k1', type=int, default= 3, help='Number of different train/test splits')
    parser.add_argument('--k2', type=int, default= 2, help='Number of training seeds per split')
    parser.add_argument('--base_seed', type=int, default=2, help='Base seed for reproducibility')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    

    M = 2 # number of modalities, used for model creation and dataset generation
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

    
    
    # Create or load the *full dataset once*
    if not args.load_data:
        print(f"Load dataset not set. Creating new synthetic dataset...")
        dataset = create_dataset_synth(data_config)
        print(f"Synthetic dataset created with {len(dataset)} samples.")
        if args.save_data:
            save_path = os.path.join(script_dir, "..", "data", "repercent_synthetic", f"dataset2{M}")
            save_dataset(dataset, save_path, data_config)

    else:
        # Load the dataset
        load_path = os.path.join(script_dir, "..", "data", "repercent_synthetic", f"dataset2{M}")
        dataset = torch.load(os.path.join(load_path, "dataset.pt"), weights_only=False)
    
    
    group_name = time.strftime("%Y-%m-%d_%H-%M-%S")
    group_name += f"_{args.model_type}_splits_{args.k1}_seeds_{args.k2}" if args.model_type == "repercent" else f"_{model_config['shared_encoder']['type']}_splits_{args.k1}_seeds_{args.k2}"
    # Initialize list to store final metrics across all runs
    all_final_metrics = []

    # Data splits
    for split_idx in range(args.k1):
        if split_idx < 2:
            continue
        split_seed = args.base_seed + 10_000 + split_idx
        
        # deterministic split
        train_dataset, test_dataset, val_dataset = split_dataset_seeded(dataset, \
                                                                        test_size=training_config["training"]["test_size"], \
                                                                        val_size= training_config["training"]["val_size"], \
                                                                        seed=split_seed)


        if args.save_data_split:
            # save split per split_idx and seed for reproducibility
            print(f"Saving data split {split_idx}...")
            save_path = os.path.join(script_dir, "..", "data", "repercent_synthetic", f"dataset2{M}")
            save_data_split(train_dataset, test_dataset, val_dataset= val_dataset, save_path= save_path, split_id= str(split_idx))
        
        
        # Seeds per split - model initialization and training
        for seed_idx in range(args.k2):
            
            train_seed = args.base_seed + 100 * split_idx + seed_idx
            set_seed(train_seed)
            generator = torch.Generator().manual_seed(train_seed)
            # dataloaders
            train_loader, test_loader, val_loader = make_dataloaders(train_dataset, test_dataset, val_dataset= val_dataset, batch_size=training_config["training"]["batch_size"], generator=generator)

            # Initialize wandb run and log hyperparameters
            run = wandb.init(
                project=model_config["wandb"]["project"],
                group=group_name,
                name=f"{group_name}_split_{split_idx}_seed_{seed_idx}",
                config={
                    "k1": args.k1, "k2": args.k2, "base_seed": args.base_seed,
                    "model_type": args.model_type,
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
                                disen_mapping=model_config["repercent"]["disen_mapping"]).to(device)

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
            run_key = f"split{split_idx}_seed{seed_idx}"

            # Logging identifiers
            wandb.log({
                "meta/split_idx": split_idx,
                "meta/seed_idx": seed_idx,
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
                "seed_idx": seed_idx,
                "split_seed": split_seed,
                "train_seed": train_seed,
                "metrics": final_metrics,
            })
            wandb.finish()
    return
    
    # global summary run
    run = wandb.init(project= model_config["wandb"]["project"], 
                    group=group_name, name= f"aggregate_{args.model_type}" if args.model_type == "repercent" else f"aggregate_{model_config['shared_encoder']['type']}", 
                    reinit=True, config={"k1": args.k1, "k2": args.k2, "base_seed": args.base_seed, "model_type": args.model_type})
    if args.log_dataset_artifact:
        log_dataset(
            dataset_name=f"dataset2{M}",
            dataset_path=os.path.join(script_dir, "..", "data", "repercent_synthetic"),
            data_config_path=data_config_path
        )
    aggregate_and_log(all_final_metrics)
    wandb.finish()

if __name__ == "__main__":
    main()