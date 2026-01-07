import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from typing import Literal, List
from torch.utils.data import DataLoader
from src.utils.synthetic_dataset_2m import GenerateData, MultimodalDataset, save_dataset, save_data_split
from src.models.perceiver import Perceiver
from src.models.repercent_2m import DisenEncoder, RePercENT, DisenLoss
from training.train_repercent_2m import split_dataset, make_dataloaders, train, make_model
from training.log_data import log_model_details, log_model_checkpoint, log_dataset
from training.train_jointopt_2m import make_model_jointopt
from src.models.jointopt_2m import JointOpt
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

def set_seed(seed: int):
    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch (CPU & GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For CUDA >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def create_dataset_synth(data_config):
    '''
    Create synthetic dataset based on the data configuration and save it to the specified path.
    Args:
        data_config: Configuration dictionary for the data.
    '''
    gen_data = GenerateData(N_data= data_config["create_data"]["N_data"], trans_type= data_config["create_data"]["trans_type"], latent_dims= data_config["create_data"]["latent_dims"])
    gen_data.create_dataset(dist= data_config["create_data"]["dist"], t1= data_config["create_data"]["t1"], t2= data_config["create_data"]["t2"], gamma1= data_config["create_data"]["gamma1"], gamma2= data_config["create_data"]["gamma2"], normalize= data_config["create_data"]["normalize"], sigmas= data_config["create_data"]["sigmas"])
    dataset = MultimodalDataset(total_data= gen_data.dataset_dict['total_data'], labels_1= gen_data.dataset_dict['labels_1'], labels_2= gen_data.dataset_dict['labels_2'], labels_s= gen_data.dataset_dict['labels_s'])

    return dataset


def main():
    set_seed(0)
    
    parser = argparse.ArgumentParser(description="Train RePercENT model on synthetic data")
    parser.add_argument('--save_data', type=bool, default=True, help='Whether to save the created dataset')
    parser.add_argument('--save_data_split', type=bool, default=True, help='Whether to save the train-test data split')
    parser.add_argument('--load_data', type=bool, default=False, help='Whether to load an existing dataset')
    parser.add_argument('--log_dataset_artifact', type=bool, default=True, help='Whether to log the dataset as a W&B artifact')
    parser.add_argument('--model_type', type= str, choices=['jointopt', 'repercent'], default='repercent', help='Type of model to train: jointopt or repercent')
    args = parser.parse_args()

    # device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Read the configuration files for data
    data_config_path = os.path.join(script_dir, "..", "configs", "data", "synthetic_data.yaml")
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)
    # Read the configuration files for the model
    model_config_path = os.path.join(script_dir, "..", "configs", "model", f"{args.model_type}.yaml")
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    # Read the configuration files for training
    training_config_path = os.path.join(script_dir, "..", "configs", "training", "train_synthetic.yaml")
    with open(training_config_path, 'r') as f:
        training_config = yaml.safe_load(f)

    
    # Create the dataset based on the data configuration
    if not args.load_data:
        dataset = create_dataset_synth(data_config)
        
        if args.save_data:
            # create directory if it doesn't exist
            save_path = os.path.join(script_dir, "..", "data", "repercent_synthetic", "dataset14")
            save_dataset(dataset, save_path, data_config)
        
        # split dataset into train and test
        train_dataset, test_dataset = split_dataset(dataset, test_size= training_config["training"]["test_size"])
        

        if args.save_data_split:
            # save the train and test splits
            save_path = os.path.join(script_dir, "..", "data", "repercent_synthetic", "dataset14")
            save_data_split(train_dataset, test_dataset, save_path)

    else:
        # load train and test datasets from artifact
        load_path = os.path.join(script_dir, "..", "data", "repercent_synthetic", "dataset14")
        split_data = torch.load(os.path.join(load_path, "data_split.pt"), weights_only=False)
        train_dataset = split_data['train_dataset']
        test_dataset = split_data['test_dataset']

    train_loader, test_loader = make_dataloaders(train_dataset, test_dataset, batch_size= training_config["training"]["batch_size"])

    if args.model_type == 'jointopt':
        model = make_model_jointopt(model_config, data_config).to(device)
    elif args.model_type == 'repercent':
        # Define the disentangled encoders
        disen_m1 = make_model(model_config, data_config, modality='m1')
        disen_m2 = make_model(model_config, data_config, modality='m2')

        # Define the RePercENT model
        model= RePercENT(M=2, disenEncoder= [disen_m1, disen_m2]).to(device)


    # 2. Initialize W&B
    run = wandb.init(project= data_config["wandb"]["project"], name= time.strftime("%Y-%m-%d_%H-%M-%S") + f"_{args.model_type}")

    if args.log_dataset_artifact:
        # log dataset to wandb
        log_dataset(
            dataset_name= "dataset13",
            dataset_path= os.path.join(script_dir, "..", "data", "repercent_synthetic"),
            data_config_path= data_config_path
        )

    # Log the model, data, and training configurations to W&B
    log_model_details(
        run,
        model_name= args.model_type,
        data_config= data_config_path,
        model_config= model_config_path,
        training_config= training_config_path
    )
    

    # 3. Training model
    disen_loss = DisenLoss(alpha= training_config["disen_loss"]["alpha"], lmd=training_config["disen_loss"]["lmd"], lmd_end_value= training_config["disen_loss"]["lmd_end_value"])
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config["optimizer"]["lr"], weight_decay= training_config["optimizer"]["weight_decay"])
    train(train_loader, test_loader, model, optimizer, disen_loss, training_config["training"]["n_epochs"], device, checkpoint_dir= os.path.join(script_dir, '..', 'checkpoints', 'repercent_synthetic', run.name))


    # 6. Finish W&B run
    wandb.finish()

if __name__ == "__main__":
    main()