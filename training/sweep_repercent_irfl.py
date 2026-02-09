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
from training.train_irfl import train_sweep
from src.utils.irfl_dataset import make_dataset
from src.utils.helpers import set_seed
from training.log_data import log_model_details, log_model_checkpoint
import math
import numpy as np
import yaml
import argparse
import time


import wandb
import random


# globals
TRAIN_DATASET = None
TRAIN_DICT = None

def load_datasets_once(args, data_config, script_dir):
    global TRAIN_DATASET, TRAIN_DICT
    if TRAIN_DATASET is not None:
        return  # already loaded

    total_train_data = torch.load(os.path.join(script_dir, args.datasets_path, "IRFL_train_tensors.pt"), map_location="cpu")
    
    total_train_data_aug = torch.load(os.path.join(script_dir, args.datasets_path, "IRFL_train_tensors_aug.pt"), map_location="cpu")
    

    TRAIN_DATASET, TRAIN_DICT = make_dataset(
        total_data=total_train_data | total_train_data_aug,
        num_modalities=data_config["create_data"]["M"],
        data_type="train",
        include_original=True,
    )
    

def sweep_run(args, data_config, model_config, training_config, script_dir):
    run = wandb.init(project=args.project)
    cfg = wandb.config

    seed = int(getattr(cfg, "seed", 0))
    set_seed(seed)

    # use globals
    train_dataset = TRAIN_DATASET
    

    g = torch.Generator().manual_seed(seed)
    temp_train_dataset, temp_val_dataset = split_dataset(train_dataset, test_size=0.1, generator=g)

    batch_size = int(getattr(cfg, "batch_size", training_config["training"]["batch_size"]))
    train_loader, val_loader = make_dataloaders(temp_train_dataset, temp_val_dataset, batch_size=batch_size, generator=g)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hyperparams from sweep
    lr = float(getattr(cfg, "lr", training_config["optimizer"]["lr"]))
    wd = float(getattr(cfg, "weight_decay", training_config["optimizer"]["weight_decay"]))
    n_epochs = int(getattr(cfg, "n_epochs", training_config["training"]["n_epochs"]))

    alpha = float(getattr(cfg, "alpha", training_config["disen_loss"]["alpha"]))
    lmd = float(getattr(cfg, "lmd", training_config["disen_loss"]["lmd"]))
    lmd_end_value = float(getattr(cfg, "lmd_end_value", training_config["disen_loss"]["lmd_end_value"]))
    model_config["perceiver"]["cross_heads"] = int(getattr(cfg, "cross_heads", model_config["perceiver"]["cross_heads"]))
    model_config["perceiver"]["latent_heads"] = int(getattr(cfg, "latent_heads", model_config["perceiver"]["latent_heads"]))
    model_config["perceiver"]["depth"] = int(getattr(cfg, "depth", model_config["perceiver"]["depth"]))
    model_config["perceiver"]["weight_tie_layers"] = bool(getattr(cfg, "weight_tie_layers", model_config["perceiver"]["weight_tie_layers"]))


    disenEncoders = [
        make_model(model_config, data_config, modality=m + 1, M=data_config["create_data"]["M"])
        for m in range(data_config["create_data"]["M"])
    ]
    model = RePercENT(
        M=data_config["create_data"]["M"],
        disenEncoder=disenEncoders,
        recon=training_config["disen_loss"]["recon"],
        disen_mapping=model_config["repercent"]["disen_mapping"],
    ).to(device)

    disen_loss = DisenLoss(
        alpha=alpha, lmd=lmd, lmd_end_value=lmd_end_value,
        M=data_config["create_data"]["M"],
        recon=training_config["disen_loss"]["recon"],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    final_metrics = train_sweep(
        train_loader, val_loader,
        model, optimizer, disen_loss,
        n_epochs, device
    )

    wandb.log(final_metrics)
    wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_path", type=str, default="../data/irfl/datasets/")
    parser.add_argument("--model_type", type=str, choices=["repercent"], default="repercent")
    parser.add_argument("--project", type=str, default="IRFL-RePercENT-Sweeps")
    parser.add_argument("--sweep_id", type=str, default= "irfl_sweep_1", help="The sweep config file name without .yaml extension")
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    M = 3

    # load configs once
    data_config_path = os.path.join(script_dir, "..", "configs", "data", f"irfl_data_{M}m.yaml")
    model_config_path = os.path.join(script_dir, "..", "configs", "model", f"{args.model_type}_irfl_{M}m.yaml")
    training_config_path = os.path.join(script_dir, "..", "configs", "training", f"train_irfl_{M}m.yaml")
    sweep_config_path = os.path.join(script_dir, "..", "configs", "sweeps", f"irfl_sweep.yaml")

    with open(data_config_path, "r") as f:
        data_config = yaml.safe_load(f)

    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    with open(training_config_path, "r") as f:
        training_config = yaml.safe_load(f)

    # Load the sweep config
    with open(sweep_config_path, "r") as f:
        sweep_config = yaml.safe_load(f)

    sweep = wandb.sweep(sweep_config, project=args.project)
    # LOAD DATA ONCE HERE
    load_datasets_once(args, data_config, script_dir)

    # run multiple sweep trials WITHOUT reloading
    wandb.agent(
        sweep_id=sweep,
        function=lambda: sweep_run(args, data_config, model_config, training_config, script_dir),
        count=args.count,
    )

if __name__ == "__main__":
    main()