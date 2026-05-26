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
from training.train_jointopt import make_model_jointopt
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





def main():
    parser = argparse.ArgumentParser(description="Ablation study for different values of alpha")
    parser.add_argument('--model_type', type=str, choices=['jointopt', 'repercent'], default='repercent', help='Type of model to train')

    # Define number of splits and seeds
    parser.add_argument('--k1', type=int, default= 3, help='Number of different train/test splits')
    parser.add_argument('--base_seed', type=int, default=2, help='Base seed for reproducibility')
    parser.add_argument('--alpha_values', nargs='+', type=float, default=[0.01, 0.1, 1.0, 10.0, 100.0], help='Values of alpha to iterate over')
    parser.add_argument('--M_values', nargs='+', type=int, default=[3, 4, 5], help='Number of modalities to iterate over. The number of modalities, should correspond to the existing generated synthetic datasets.')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    for M in args.M_values: # modality numbers to iterate over
    
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

        

        # Load the dataset if it exists
        load_path = os.path.join(script_dir, "..", "data", "repercent_synthetic", f"dataset2{M}")
        try:
            dataset = torch.load(os.path.join(load_path, "dataset.pt"), weights_only=False)
        except FileNotFoundError as e:
            raise ValueError(
                f"The synthetic dataset for M = {M} does not exist. "
                f"Please first create the dataset for {M} modalities and then run the script "
                f"or change the `--M_values` argument."
            ) from e

        group_name = f"{args.model_type}_splits_{args.k1}" if args.model_type == "repercent" else f"_{model_config['shared_encoder']['type']}_splits_{args.k1}"
        group_name += f"_{M}M" # modality number identifier for wandb grouping
        
        

        for alpha in args.alpha_values: # alpha values to iterate over
       
            split_seed = args.base_seed + 10_000 + 0
            
            # deterministic split
            train_dataset, test_dataset, val_dataset = split_dataset_seeded(dataset, \
                                                                            test_size=training_config["training"]["test_size"], \
                                                                            val_size= training_config["training"]["val_size"], \
                                                                            seed=split_seed)


            train_seed = args.base_seed + 100 * 0
            set_seed(train_seed)
            generator = torch.Generator().manual_seed(train_seed)
            # dataloaders
            train_loader, test_loader, val_loader = make_dataloaders(train_dataset, test_dataset, val_dataset= val_dataset, batch_size=training_config["training"]["batch_size"], generator=generator)

            # Initialize wandb run and log hyperparameters
            run = wandb.init(
                project= "repercent_alpha_ablation_synthetic",
                group=group_name,
                name=f"{group_name}_M{M}_alpha{alpha}_split{2}",
                config={
                    "k1": args.k1, "base_seed": args.base_seed,
                    "model_type": args.model_type,
                    "alpha": alpha, "M": M
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

            disen_loss = DisenLoss(alpha= alpha,
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
            run_key = f"split{0}"

            # Logging identifiers
            wandb.log({
                "meta/split_idx": 0,
                "meta/split_seed": split_seed,
                "meta/train_seed": train_seed,
            })

            # TRAIN
            final_metrics = train(train_loader, test_loader, model, optimizer, disen_loss, \
                                training_config["training"]["n_epochs"], \
                                device, val_loader= val_loader, \
                                checkpoint_dir=os.path.join(script_dir, '..', 'checkpoints', 'repercent_synthetic', run.name, run_key), \
                                generator= generator)

            metrics_table = wandb.Table(columns=["metric", "value"])
            for metric_name, metric_value in sorted(final_metrics.items()):
                metrics_table.add_data(metric_name, float(metric_value))
            wandb.log({f"final_metrics/M{M}/alpha{alpha}": metrics_table})
            
            wandb.finish()
        
        
    

if __name__ == "__main__":
    main()
