import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import torch.nn as nn
from typing import Literal, List
from torch.utils.data import random_split
import wandb
from src.utils.helpers import extract_latents_and_labels, linear_probe, plot_confusion_matrix
from src.models.jointopt_2m import JointOpt, simpleEncoder
from training.train_repercent_2m import train_loop, test_loop
import numpy as np
import math

def make_model_jointopt(model_config_jointopt: dict, device: torch.device) -> nn.Module:
    '''
    Create JointOpt model based on the model configuration.
    Args:
        model_config (dict): Configuration dictionary for the model.
        device (torch.device): Device to load the model onto.
    Returns:
        JointOpt: Instantiated JointOpt model.
    '''
    # Shared Encoders
    input_dims_shared = model_config_jointopt["shared_encoder"]["input_dims"]
    hidden_dims_shared = model_config_jointopt["shared_encoder"]["hidden_dims"]
    output_dims_shared = model_config_jointopt["shared_encoder"]["latent_dims"]
    activation_shared = model_config_jointopt["shared_encoder"]["activation"]

    sharedEncoders = []
    for (input_dim, hidden_dims, output_dim) in zip(input_dims_shared, hidden_dims_shared, output_dims_shared):
        encoder = simpleEncoder(input_dim= input_dim, hidden_dims= hidden_dims, latent_dim= output_dim, activation= activation_shared)
        sharedEncoders.append(encoder)

    # Unique Encoders
    input_dims_unique = model_config_jointopt["unique_encoder"]["input_dims"]
    hidden_dims_unique = model_config_jointopt["unique_encoder"]["hidden_dims"]
    output_dims_unique = model_config_jointopt["unique_encoder"]["latent_dims"]
    activation_unique = model_config_jointopt["unique_encoder"]["activation"]

    uniqueEncoders = []
    for (input_dim, hidden_dims, output_dim) in zip(input_dims_unique, hidden_dims_unique, output_dims_unique):
        encoder = simpleEncoder(input_dim= input_dim, hidden_dims= hidden_dims, latent_dim= output_dim, activation= activation_unique)
        uniqueEncoders.append(encoder)


    model = JointOpt(M= len(sharedEncoders), 
                    sharedEncoders= sharedEncoders, 
                    uniqueEncoders= uniqueEncoders, 
                    vmfkappa= model_config_jointopt["vmfkappa"], 
                    add_shared= model_config_jointopt["add_shared"])

    return model