import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from typing import Literal, List
from torch.utils.data import DataLoader
from src.utils.synthetic_dataset import GenerateData, MultimodalDataset, save_dataset, save_data_split
from src.models.perceiver import Perceiver
from src.models.repercent import DisenEncoder, RePercENT, DisenLoss
from training.train_repercent import make_dataloaders, train
import math
from tqdm.notebook import tqdm
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import yaml
import argparse
import time

import wandb


def make_model(model_config, data_config, modality: Literal['m1', 'm2']):
    enc_m = nn.Identity()

    DEPTH = model_config["perceiver"]["depth"]
    if modality == 'm2':
        MAX_FREQ = math.ceil(data_config["create_data"]["t2"]/ 2) if model_config["perceiver"]["max_freq"] is None else model_config["perceiver"]["max_freq"]
    else:
        MAX_FREQ = math.ceil(data_config["create_data"]["t1"]/ 2) if model_config["perceiver"]["max_freq"] is None else model_config["perceiver"]["max_freq"]
    NUM_FREQ_BANDS= math.floor(math.log2(MAX_FREQ)) + 1 if model_config["perceiver"]["num_freq_bands"] is None else model_config["perceiver"]["num_freq_bands"]
    if modality == 'm2':
        INPUT_CHANNELS= data_config["create_data"]["latent_dims"]["Z2"] + data_config["create_data"]["latent_dims"]["Zs"] if model_config["perceiver"]["input_channels"] is None else model_config["perceiver"]["input_channels"]
    else:
        INPUT_CHANNELS= data_config["create_data"]["latent_dims"]["Z1"] + data_config["create_data"]["latent_dims"]["Zs"] if model_config["perceiver"]["input_channels"] is None else model_config["perceiver"]["input_channels"]
    INPUT_AXIS= model_config["perceiver"]["input_axis"]
    LATENT_DIM= model_config["perceiver"]["latent_dim"]
    NUM_LATENTS= model_config["perceiver"]["num_latents"]
    CROSS_HEADS= model_config["perceiver"]["cross_heads"]
    LATENT_HEADS= model_config["perceiver"]["latent_heads"]
    POS_ENCODING= model_config["perceiver"]["pos_encoding"]

    
    per_m = Perceiver(num_freq_bands= NUM_FREQ_BANDS,
                        latent_dim= LATENT_DIM,
                        num_latents= NUM_LATENTS,
                        depth= DEPTH,
                        max_freq= MAX_FREQ,
                        latent_heads= LATENT_HEADS,
                        cross_heads= CROSS_HEADS,
                        input_channels= INPUT_CHANNELS,
                        input_axis= INPUT_AXIS,
                        fourier_encode_data= POS_ENCODING)

    disen_m = DisenEncoder(encoder_model= enc_m, perceiver_model= per_m)

    return disen_m

def main():
    parser = argparse.ArgumentParser(description="Train RePercENT model on synthetic data")
    parser.add_argument('--save_data', type=bool, default=True, help='Whether to save the created dataset')
    parser.add_argument('--save_data_split', type=bool, default=True, help='Whether to save the train-test data split')
    args = parser.parse_args()

    # device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Read the configuration files for data
    data_config_path = os.path.join(script_dir, "..", "configs", "data", "synthetic_data.yaml")
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)
    # Read the configuration files for the model
    model_config_path = os.path.join(script_dir, "..", "configs", "model", "repercent.yaml")
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    # Read the configuration files for training
    training_config_path = os.path.join(script_dir, "..", "configs", "training", "train_synthetic.yaml")
    with open(training_config_path, 'r') as f:
        training_config = yaml.safe_load(f)

    
    # Create the dataset based on the data configuration
    gen_data = GenerateData(N_data= data_config["create_data"]["N_data"], mod_type= data_config["create_data"]["mod_type"], latent_dims= data_config["create_data"]["latent_dims"])
    gen_data.create_dataset(t1= data_config["create_data"]["t1"], t2= data_config["create_data"]["t2"], gamma1= data_config["create_data"]["gamma1"], gamma2= data_config["create_data"]["gamma2"], normalize= data_config["create_data"]["normalize"])
    dataset = MultimodalDataset(total_data= gen_data.dataset_dict['total_data'], labels_1= gen_data.dataset_dict['labels_1'], labels_2= gen_data.dataset_dict['labels_2'], labels_s= gen_data.dataset_dict['labels_s'])

    if args.save_data:
        # create directory if it doesn't exist
        save_path = os.path.join(script_dir, "..", "data", "repercent_synthetic", "dataset5")
        save_dataset(dataset, save_path, data_config)
        
    # Define the disentangled encoders
    disen_m1 = make_model(model_config, data_config, modality='m1')
    disen_m2 = make_model(model_config, data_config, modality='m2')

    # Define the RePercENT model
    model= RePercENT(M=2, disenEncoder= [disen_m1, disen_m2]).to(device)
    
    # 2. Initialize W&B
    run = wandb.init(project= data_config["wandb"]["project"], name= time.strftime("%Y%m%d-%H%M%S") + "_repercent_synthetic")

    #make dataloaders
    train_loader, test_loader = make_dataloaders(dataset, training_config["training"])

    if args.save_data_split:
        # save the train and test splits
        train_data = train_loader.dataset
        test_data = test_loader.dataset
        save_path = os.path.join(script_dir, "..", "data", "repercent_synthetic", "dataset5")
        save_data_split(train_data, test_data, save_path)
        

    disen_loss = DisenLoss(alpha= training_config["disen_loss"]["alpha"], lmd=training_config["disen_loss"]["lmd"], lmd_end_value= training_config["disen_loss"]["lmd_end_value"])
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config["optimizer"]["lr"], weight_decay= training_config["optimizer"]["weight_decay"])
    train(gen_data, train_loader, test_loader, model, optimizer, disen_loss, training_config["training"]["n_epochs"], device, checkpoint_dir= os.path.join(script_dir, '..', 'checkpoints', 'repercent_synthetic', run.name))


    # 6. Finish W&B run
    wandb.finish()

if __name__ == "__main__":
    main()