import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import torch.nn as nn
from typing import Literal, List
from torch.utils.data import random_split
import wandb
from src.models.perceiver import Perceiver
from src.models.repercent_2m import DisenEncoder, RePercENT, DisenLoss
from src.utils.synthetic_dataset_2m import GenerateData
from src.utils.helpers import extract_latents_and_labels, linear_probe, plot_confusion_matrix
from training.log_data import log_model_checkpoint
import matplotlib.pyplot as plt
import numpy as np
import math

def split_dataset(dataset, test_size: float):
    train_size = int((1 - test_size) * len(dataset))
    test_size = int(test_size * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

def make_dataloaders(train_dataset, test_dataset, batch_size: int= 16):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= batch_size, shuffle= True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size= batch_size, shuffle= False)
    return train_loader, test_loader

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
                        weight_tie_layers= WEIGHT_TIE_LAYERS)

    disen_m = DisenEncoder(encoder_model= enc_m, perceiver_model= per_m)

    return disen_m

def train_loop(data_m1, data_m2, data_m1_aug, data_m2_aug, model, optimizer, disen_loss):
    """
    Single Epoch training step for RePercENT model
    Args:
        data_m1: Batch data from modality 1
        data_m2: Batch data from modality 2
        data_m1_aug: Augmented batch data from modality 1
        data_m2_aug: Augmented batch data from modality 2
        model: RePercENT model in training mode
        optimizer: Optimizer for RePercENT model
        disen_loss: Disentanglement loss function
    Returns:
        loss: Computed loss value for the batch
        logs: Dictionary containing loss components for monitoring
    """
    # Forward pass through RePercENT
    outputs = model(data_m1, data_m2)
    outputs_aug = model(data_m1_aug, data_m2_aug)
    
    # Compute disentanglement loss
    loss, logs = disen_loss(outputs, outputs_aug)
    
    # Backward pass for RePercENT
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss, logs
    

def test_loop(data_m1, data_m2, data_m1_aug, data_m2_aug, model, disen_loss):
    """
    Single Epoch testing step for RePercENT model
    Args:
        data_m1: Batch data from modality 1
        data_m2: Batch data from modality 2
        data_m1_aug: Augmented batch data from modality 1
        data_m2_aug: Augmented batch data from modality 2
        model: RePercENT model in evaluation mode
        disen_loss: Disentanglement loss function
    Returns:
        loss: Computed loss value for the batch
        logs: Dictionary containing loss components for monitoring
    """
    # Forward pass through RePercENT
    outputs = model(data_m1, data_m2)
    outputs_aug = model(data_m1_aug, data_m2_aug)
    
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
    # Training loop
    for _iter in range(epochs):
        epoch_loss = 0.0
        epoch_ortho_loss = 0.0
        epoch_unique_loss = 0.0
        epoch_shared_loss = 0.0
        
        model.train()
        
        print(f"----- Epoch: {_iter + 1} / {epochs} -----")
        
        # Training phase
        for batch_idx, (data_m1, data_m2, _, _, _) in enumerate(train_loader):
            temp_b = data_m1.shape[0]
            data_m1 = data_m1.to(device)
            data_m2 = data_m2.to(device)
            
            # print(f"M1: {data_m1[0, 0, -3:]}, M2: {data_m2[0, 0, -3:]}")
            # Augment data 
            data_m1_aug = GenerateData.augment_data(data_m1, aug_type="random")
            data_m2_aug = GenerateData.augment_data(data_m2, aug_type="random")
            
            # Forward pass through RePercENT
            loss_train, logs_train = train_loop(data_m1, data_m2, data_m1_aug, data_m2_aug, model, optimizer, disen_loss)
            
            # Track losses
            epoch_loss += loss_train.item()/ temp_b
            epoch_ortho_loss += logs_train["ortho"]/ temp_b
            epoch_unique_loss += logs_train["unique"]/ temp_b
            epoch_shared_loss += logs_train["shared"]/ temp_b
        
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
            
            for batch_idx, (data_m1, data_m2, _, _, _) in enumerate(val_loader):
                temp_b = data_m1.shape[0]
                data_m1 = data_m1.to(device)
                data_m2 = data_m2.to(device)
                
                # Augment data 
                data_m1_aug = GenerateData.augment_data(data_m1, aug_type="random")
                data_m2_aug = GenerateData.augment_data(data_m2, aug_type="random")
                
                # Forward pass through RePercENT
                loss_val, logs_val = test_loop(data_m1, data_m2, data_m1_aug, data_m2_aug, model, disen_loss)
                
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
        train_data_dict = extract_latents_and_labels(model, train_loader, device)
        val_data_dict = extract_latents_and_labels(model, val_loader, device)

        # Calculate Linear Probe accuracies
        linear_probe_acc = {"u_12": np.zeros(3), "u_21": np.zeros(3), "s": np.zeros(3)}
        for i, label in enumerate(['labels_1', 'labels_2', 'labels_s']):
            # Unique component of modality 1
            linear_probe_acc["u_12"][i] = linear_probe(
                train_data_dict['u_12'], train_data_dict[label],
                val_data_dict['u_12'], val_data_dict[label]
            )
            # Unique component of modality 2
            linear_probe_acc["u_21"][i] = linear_probe(
                train_data_dict['u_21'], train_data_dict[label],
                val_data_dict['u_21'], val_data_dict[label]
            )
            # Shared component from modality 2
            linear_probe_acc["s"][i] = linear_probe(
                np.concatenate((train_data_dict['s_21'], train_data_dict['s_12']), axis= -1), train_data_dict[label],
                np.concatenate((val_data_dict['s_21'], val_data_dict['s_12']), axis= -1), val_data_dict[label]
            )

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
            "probe/u_12/labels_1/acc": linear_probe_acc["u_12"][0],
            "probe/u_12/labels_2/acc": linear_probe_acc["u_12"][1],
            "probe/u_12/labels_s/acc": linear_probe_acc["u_12"][2],
            "probe/u_21/labels_2/acc": linear_probe_acc["u_21"][1],
            "probe/u_21/labels_1/acc": linear_probe_acc["u_21"][0],
            "probe/u_21/labels_s/acc": linear_probe_acc["u_21"][2],
            "probe/shared/labels_1/acc": linear_probe_acc["s"][0],
            "probe/shared/labels_2/acc": linear_probe_acc["s"][1],
            "probe/shared/labels_s/acc": linear_probe_acc["s"][2],
            "confusion_matrix": wandb.Image(plot_confusion_matrix(linear_probe_acc))
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
    
    print("Training complete!")