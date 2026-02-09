import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import torch.nn as nn
from typing import Literal, List
from torch.utils.data import random_split
import wandb
from src.models.perceiver import Perceiver
from src.models.repercent import DisenEncoder, RePercENT, DisenLoss
from src.utils.synthetic_dataset import GenerateData
from src.utils.helpers import ProbeEvaluator, extract_latents_and_labels, linear_probe, regression_probe, plot_confusion_matrix, plot_pairwise_confusion_matrices
from training.log_data import log_model_checkpoint
import matplotlib.pyplot as plt
import numpy as np
import math
from itertools import combinations



def split_dataset(dataset, test_size: float, generator: torch.Generator= None):
    train_size = int((1 - test_size) * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator= generator)
    return train_dataset, test_dataset

def make_dataloaders(train_dataset, test_dataset, batch_size: int= 16, generator: torch.Generator= None, shuffle_train: bool= True, shuffle_test: bool= False):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= batch_size, shuffle= shuffle_train, generator=generator)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size= batch_size, shuffle= shuffle_test, generator=generator)
    return train_loader, test_loader

def make_model(model_config, data_config, modality: int= 2, M: int=2):
    """
    Create a single DisenEncoder model for a given modality based on the model and data configurations.
    Args:
        model_config: Configuration dictionary for the model.
        data_config: Configuration dictionary for the data.
        modality: Modality number (1, 2, ..., M) for which the DisenEncoder is to be created. NOTE: use 1-based indexing.
        M: Total number of modalities.
    Returns:
        disen_m: DisenEncoder model for the specified modality.
    """
    enc_m = nn.Identity()

    DEPTH = model_config["perceiver"]["depth"]
   
    MAX_FREQ = math.ceil(data_config["create_data"]["ts"][modality - 1]/ 2) if model_config["perceiver"]["max_freq"] is None else model_config["perceiver"]["max_freq"]
    NUM_FREQ_BANDS= math.floor(math.log2(MAX_FREQ)) + 1 if model_config["perceiver"]["num_freq_bands"] is None else model_config["perceiver"]["num_freq_bands"]
    INPUT_CHANNELS= 2**(M - 1) *data_config["create_data"]["latent_dim"] if model_config["perceiver"]["input_channels"] is None else model_config["perceiver"]["input_channels"][modality - 1]
    INPUT_AXIS= model_config["perceiver"]["input_axis"]
    LATENT_DIM= model_config["perceiver"]["latent_dim"]
    NUM_LATENTS= model_config["perceiver"]["num_latents"]
    CROSS_HEADS= model_config["perceiver"]["cross_heads"]
    LATENT_HEADS= model_config["perceiver"]["latent_heads"]
    POS_ENCODING= model_config["perceiver"]["pos_encoding"]
    WEIGHT_TIE_LAYERS= model_config["perceiver"]["weight_tie_layers"]

    
    per_m = Perceiver(num_freq_bands= NUM_FREQ_BANDS,
                        latent_dim= LATENT_DIM,
                        num_latents= NUM_LATENTS,
                        depth= DEPTH,
                        max_freq= MAX_FREQ,
                        latent_heads= LATENT_HEADS,
                        cross_heads= CROSS_HEADS,
                        input_channels= INPUT_CHANNELS,
                        input_axis= INPUT_AXIS,
                        fourier_encode_data= POS_ENCODING,
                        weight_tie_layers= WEIGHT_TIE_LAYERS,
                        use_moeffn= model_config["perceiver"].get("use_moeffn", False)
                        )
    print(f"input channels: {INPUT_CHANNELS}, latent dim: {LATENT_DIM}, num latents: {NUM_LATENTS}")
    disen_m = DisenEncoder(encoder_model= enc_m, perceiver_model= per_m)

    return disen_m


def train_loop(X, X_aug, model, optimizer, disen_loss):
    """
    Single Epoch training step for RePercENT model
    Args:
        X: Batch data from all modalities
        X_aug: Augmented batch data from all modalities
        model: RePercENT model in training mode
        optimizer: Optimizer for RePercENT model
        disen_loss: Disentanglement loss function
    Returns:
        loss: Computed loss value for the batch
        logs: Dictionary containing loss components for monitoring
    """
    # Forward pass through RePercENT
    outputs = model(X)
    outputs_aug = model(X_aug)
    
    # Compute disentanglement loss
    loss, logs = disen_loss(outputs, outputs_aug)
    
    # Backward pass for RePercENT
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss, logs
    

def test_loop(X, X_aug, model, disen_loss):
    """
    Single Epoch testing step for RePercENT model
    Args:
        X: Batch data from all modalities
        X_aug: Augmented batch data from all modalities
        model: RePercENT model in evaluation mode
        disen_loss: Disentanglement loss function
    Returns:
        loss: Computed loss value for the batch
        logs: Dictionary containing loss components for monitoring
    """
    # Forward pass through RePercENT
    outputs = model(X)
    outputs_aug = model(X_aug)
    
    # Compute disentanglement loss
    loss, logs = disen_loss(outputs, outputs_aug)
    
    return loss, logs



def train(train_loader, val_loader, model, optimizer, disen_loss, epochs, device, checkpoint_dir="./checkpoints"):
    """
    Full training loop for RePercENT model with evaluation on test set
    Args:
        train_loader: DataLoader for training dataset
        val_loader: DataLoader for validation dataset
        model: RePercENT model (This training function also supports the JointOpt model)
        optimizer: Optimizer for RePercENT model
        disen_loss: Disentanglement loss function
        epochs: Number of training epochs
        device: Device to run the training on (CPU/GPU)
        checkpoint_dir: Directory to save model checkpoints
    """
    # clear memory
    torch.cuda.empty_cache()
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Watch model with WandB
    wandb.watch(model, log="gradients")
    print(f'Number of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    components = None
    M = disen_loss.M # number of modalities
    pairs = list(combinations(range(M), 2))
    
    evaluator = ProbeEvaluator(linear_probe= linear_probe, regression_probe= regression_probe)
    for _iter in range(epochs):
        # Initialize epoch loss trackers
        epoch_loss = 0.0
        epoch_ortho_loss = 0.0
        epoch_unique_loss = 0.0
        epoch_shared_loss = 0.0
        
        model.train()
        print(f"----- Epoch: {_iter + 1} / {epochs} -----")
        # Training phase
        for batch_idx, (X, labels_u, labels_s, _, _) in enumerate(train_loader):
            temp_b = X[0].shape[0]
            X = [X[m].to(device) for m in range(M)]

            # Augment data
            X_aug = [GenerateData.augment_data(X[m], aug_type="random") for m in range(M)]
            
            loss, loss_logs = train_loop(X, X_aug, model, optimizer, disen_loss)

            # Track losses
            epoch_loss += loss.item() / temp_b
            epoch_ortho_loss += loss_logs['ortho'] / temp_b
            epoch_unique_loss += loss_logs['unique'] / temp_b
            epoch_shared_loss += loss_logs['shared'] / temp_b
            
        # Epoch statistics
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_ortho_loss = epoch_ortho_loss / len(train_loader)
        avg_unique_loss = epoch_unique_loss / len(train_loader)
        avg_shared_loss = epoch_shared_loss / len(train_loader)

        # Calculate loss on test set
        model.eval()
        with torch.no_grad():
            val_epoch_loss = 0.0
            val_epoch_ortho_loss = 0.0
            val_epoch_unique_loss = 0.0
            val_epoch_shared_loss = 0.0
        
            for batch_idx, (X, labels_u, labels_s, _, _) in enumerate(val_loader):
                temp_b = X[0].shape[0]
                X = [X[m].to(device) for m in range(M)]
                
                # Augment data 
                X_aug = [GenerateData.augment_data(X[m], aug_type="random") for m in range(M)]
                
                # Forward pass through RePercENT
                loss_val, logs_val = test_loop(X, X_aug, model, disen_loss)
                
                # Track losses
                val_epoch_loss += loss_val.item()/ temp_b
                val_epoch_ortho_loss += logs_val["ortho"]/ temp_b
                val_epoch_unique_loss += logs_val["unique"]/ temp_b
                val_epoch_shared_loss += logs_val["shared"]/ temp_b
        
        # Epoch statistics
        avg_epoch_loss_val = val_epoch_loss / len(val_loader)
        avg_ortho_loss_val = val_epoch_ortho_loss / len(val_loader)
        avg_unique_loss_val = val_epoch_unique_loss / len(val_loader)
        avg_shared_loss_val = val_epoch_shared_loss / len(val_loader)
        

        print(f"Training  Loss(x 100): {avg_epoch_loss* 100:.5f} | Ortho (x 100): {avg_ortho_loss* 100:.5f} | Unique (x 100): {avg_unique_loss* 100:.5f} | Shared (x 100): {avg_shared_loss* 100:.5f} | Lmd: {disen_loss.lmd:.6f}, alpha: {disen_loss.alpha:.6f}")
        print(f"Testing  Loss(x 100): {avg_epoch_loss_val* 100:.5f} | Ortho (x 100): {avg_ortho_loss_val* 100:.5f} | Unique (x 100): {avg_unique_loss_val* 100:.5f} | Shared (x 100): {avg_shared_loss_val* 100:.5f} ")
        
        # Evaluate linear probe accuracy of the model's learned representations after each epoch
        train_data_dict = extract_latents_and_labels(model, train_loader, device)
        val_data_dict = extract_latents_and_labels(model, val_loader, device)

        if components is None: # set components only once - same for all epochs
            components = list(train_data_dict['Labels_U'].keys()) + list(train_data_dict['Labels_S'].keys())

        
        # Log metrics to WandB
        wandb.log({
            "train/loss": avg_epoch_loss,
            "train/loss/ortho": avg_ortho_loss,
            "train/loss/unique": avg_unique_loss,
            "train/loss/shared": avg_shared_loss,
            "val/loss": avg_epoch_loss_val,
            "val/loss/ortho": avg_ortho_loss_val,
            "val/loss/unique": avg_unique_loss_val,
            "val/loss/shared": avg_shared_loss_val,
        }, step= _iter + 1)
        plt.close("all")

        # Save model checkpoint every 10 epochs and at the end
        if (_iter + 1) % 10 == 0 or (_iter + 1) == epochs:
            checkpoint_name = f"checkpoint_epoch_{_iter + 1}.pt" if (_iter + 1) // 10 != (epochs // 10) else f"final_checkpoint.pt"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            os.makedirs(checkpoint_dir, exist_ok=True) # ensure directory exists
            # Create the state dictionary
            checkpoint = {
                'epoch': _iter + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            
            # Save locally
            torch.save(checkpoint, checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")
            
            log_model_checkpoint(wandb.run, checkpoint_path, epoch= _iter + 1)
        
        # Save final metrics:
        if (_iter + 1) == epochs:
            evaluator.set_data(train_data_dict= train_data_dict, val_data_dict= val_data_dict, M= M)
            linear_results = evaluator.calculate_linear_probe()
            reg_results = evaluator.calculate_reg_probe()

            metrics_summary = evaluator.mean_metrics(linear_results, reg_results, M= M)

            # Log the complete confusion matrix for each epoch
            wandb.log({"confusion_matrix": wandb.Image(plot_confusion_matrix(linear_results["acc"], components= components, labels= components))})
            # Log the pairwise confusion matrices, i.e. M* (M -1)/ 2 matrices for M modalities
            wandb.log({"pairwise_confusion_matrices": wandb.Image(plot_pairwise_confusion_matrices(linear_probe_acc= linear_results["acc"], \
                                                                                        M= M, \
                                                                                        components= components, \
                                                                                        pairs= pairs))})    
            # Log final metrics table
            table = wandb.Table(columns=["metric", "value"])
            for k, v in metrics_summary.items():
                table.add_data(k, float(v))
            wandb.log({"final_metrics": table})
    
    print("Training complete!")
    return metrics_summary