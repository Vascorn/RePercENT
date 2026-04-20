import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import torch.nn as nn
from typing import Literal, List
from torch.utils.data import random_split
import wandb
from src.models.perceiver import Perceiver, PerceiverDisen
from src.models.repercent import DisenEncoder, RePercENT, DisenLoss
from src.utils.synthetic_dataset import GenerateData
from src.utils.helpers import ProbeEvaluator, extract_latents_and_labels, linear_probe, non_linear_probe, regression_probe, plot_confusion_matrix, plot_pairwise_confusion_matrices
from training.log_data import log_model_checkpoint
import matplotlib.pyplot as plt
import numpy as np
import math
from itertools import combinations
import copy



def split_dataset(dataset, test_size: float, generator: torch.Generator= None):
    train_size = int((1 - test_size) * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator= generator)
    return train_dataset, test_dataset

def make_dataloaders(train_dataset, test_dataset, val_dataset= None, batch_size: int= 16, generator: torch.Generator= None, shuffle_train: bool= True, shuffle_test: bool= False):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= batch_size, shuffle= shuffle_train, generator=generator)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size= batch_size, shuffle= shuffle_test, generator=generator)
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size= batch_size, shuffle= False, generator=generator)
        return train_loader, test_loader, val_loader

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
    CROSS_HEADS_DIM= model_config["perceiver"].get("cross_heads_dim", LATENT_DIM // CROSS_HEADS)
    POS_ENCODING= model_config["perceiver"]["pos_encoding"]
    WEIGHT_TIE_LAYERS= model_config["perceiver"]["weight_tie_layers"]

    perceiver_type = model_config["perceiver"].get("type", "standard")
    
    match perceiver_type:
        case "standard":
            LATENT_HEADS= model_config["perceiver"]["latent_heads"]
            LATENT_HEADS_DIM= model_config["perceiver"].get("latent_heads_dim", LATENT_DIM // LATENT_HEADS)
            per_m = Perceiver(num_freq_bands= NUM_FREQ_BANDS,
                            latent_dim= LATENT_DIM,
                            num_latents= NUM_LATENTS,
                            depth= DEPTH,
                            max_freq= MAX_FREQ,
                            latent_heads= LATENT_HEADS,
                            latent_dim_head= LATENT_HEADS_DIM,
                            cross_heads= CROSS_HEADS,
                            cross_dim_head= CROSS_HEADS_DIM,
                            input_channels= INPUT_CHANNELS,
                            input_axis= INPUT_AXIS,
                            fourier_encode_data= POS_ENCODING,
                            weight_tie_layers= WEIGHT_TIE_LAYERS,
                            use_moeffn= model_config["perceiver"].get("use_moeffn", False),
                            use_slot_attn= model_config["perceiver"].get("use_slot_attn", True)
                            )
            print(f"Created standard Perceiver with latent heads: {LATENT_HEADS}, latent head dim: {LATENT_HEADS_DIM}, cross heads: {CROSS_HEADS}, cross head dim: {CROSS_HEADS_DIM}, pos encoding: {POS_ENCODING}, weight tie layers: {WEIGHT_TIE_LAYERS}, use_moeffn: {model_config['perceiver'].get('use_moeffn', False)}, use_slot_attn: {model_config['perceiver'].get('use_slot_attn', True)}")
        case "disen":
            per_m = PerceiverDisen(num_freq_bands= NUM_FREQ_BANDS,
                            latent_dim= LATENT_DIM,
                            num_latents= NUM_LATENTS,
                            depth= DEPTH,
                            max_freq= MAX_FREQ,
                            cross_heads= CROSS_HEADS,
                            cross_dim_head= CROSS_HEADS_DIM,
                            input_channels= INPUT_CHANNELS,
                            input_axis= INPUT_AXIS,
                            fourier_encode_data= POS_ENCODING,
                            weight_tie_layers= WEIGHT_TIE_LAYERS
                            )
            print(f"input channels: {INPUT_CHANNELS}, latent dim: {LATENT_DIM}, num latents: {NUM_LATENTS}, cross heads: {CROSS_HEADS}, cross head dim: {CROSS_HEADS_DIM}, pos encoding: {POS_ENCODING}, weight tie layers: {WEIGHT_TIE_LAYERS}")

        case _:
            raise ValueError(f"Unsupported perceiver type: {perceiver_type}")

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
    outputs = model(X, mask = [None for _ in range(len(X))])
    outputs_aug = model(X_aug, mask = [None for _ in range(len(X_aug))])
    
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
    # Forward pass 
    outputs = model(X, mask = [None for _ in range(len(X))])
    outputs_aug = model(X_aug, mask = [None for _ in range(len(X_aug))])
    
    # Compute disentanglement loss
    loss, logs = disen_loss(outputs, outputs_aug)
    
    return loss, logs



def train(train_loader, test_loader, model, optimizer, disen_loss, epochs, device, val_loader= None, checkpoint_dir="./checkpoints", generator= None):
    """
    Full training loop for RePercENT model with evaluation on test set
    Args:
        train_loader: DataLoader for training dataset
        test_loader: DataLoader for test dataset.
        model: RePercENT model (This training function also supports the JointOpt model)
        optimizer: Optimizer for RePercENT model
        disen_loss: Disentanglement loss function
        epochs: Number of training epochs
        device: Device to run the training on (CPU/GPU)
        val_loader: DataLoader for validation dataset. Optional, if not provided, no validation will be performed during training and model checkpoints will be saved based on training loss.
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

    best_fw_val_loss = float('inf')
    best_model_state = None
    for _iter in range(epochs):
        # Initialize epoch loss trackers
        epoch_loss = 0.0
        epoch_ortho_loss = 0.0
        epoch_unique_loss = 0.0
        epoch_shared_loss = 0.0
        epoch_fw_loss = 0.0 # fixed weight loss for logging, when no schedulers are used, this is just the total loss with equal weights for all components
        model.train()
        print(f"----- Epoch: {_iter + 1} / {epochs} -----")
        # Training phase
        for batch_idx, (X, labels_u, labels_s, _, _) in enumerate(train_loader):
    
            X = [X[m].to(device) for m in range(M)]

            # Augment data
            X_aug = [GenerateData.augment_data(X[m], aug_type="random", generator=generator) for m in range(M)]
            
            loss, loss_logs = train_loop(X, X_aug, model, optimizer, disen_loss)

            # Track losses
            epoch_loss += loss.item()
            epoch_ortho_loss += loss_logs['ortho']
            epoch_unique_loss += loss_logs['unique']
            epoch_shared_loss += loss_logs['shared']
            epoch_fw_loss += loss_logs['fw_loss']
            
        # Epoch statistics
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_ortho_loss = epoch_ortho_loss / len(train_loader)
        avg_unique_loss = epoch_unique_loss / len(train_loader)
        avg_shared_loss = epoch_shared_loss / len(train_loader)
        avg_fw_loss = epoch_fw_loss / len(train_loader)

        print(f"Training  Loss: {avg_epoch_loss:.5f} | Ortho: {avg_ortho_loss:.5f} | Unique: {avg_unique_loss:.5f} | Shared: {avg_shared_loss:.5f} | Lmd: {disen_loss.lmd:.6f}, alpha: {disen_loss.alpha:.6f}")
        
        wandb.log({
            "train/loss": avg_epoch_loss,
            "train/loss/ortho": avg_ortho_loss,
            "train/loss/unique": avg_unique_loss,
            "train/loss/shared": avg_shared_loss,
            "train/loss/fixed_weight": avg_fw_loss,
        }, step= _iter + 1)

        # Calculate loss on validation set if provided, otherwise use training loss for checkpointing
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_epoch_loss = 0.0
                val_epoch_ortho_loss = 0.0
                val_epoch_unique_loss = 0.0
                val_epoch_shared_loss = 0.0
                val_epoch_fw_loss = 0.0
                for batch_idx, (X, labels_u, labels_s, _, _) in enumerate(val_loader):
                    temp_b = X[0].shape[0]
                    X = [X[m].to(device) for m in range(M)]
                    
                    # Augment data 
                    X_aug = [GenerateData.augment_data(X[m], aug_type="random", generator=generator) for m in range(M)]
                    
                    # Forward pass through RePercENT
                    loss_val, logs_val = test_loop(X, X_aug, model, disen_loss)
                    
                    # Track losses
                    val_epoch_loss += loss_val.item()
                    val_epoch_ortho_loss += logs_val["ortho"]
                    val_epoch_unique_loss += logs_val["unique"]
                    val_epoch_shared_loss += logs_val["shared"]
                    val_epoch_fw_loss += logs_val["fw_loss"]
            # Epoch statistics
            avg_epoch_loss_val = val_epoch_loss / len(val_loader)
            avg_ortho_loss_val = val_epoch_ortho_loss / len(val_loader)
            avg_unique_loss_val = val_epoch_unique_loss / len(val_loader)
            avg_shared_loss_val = val_epoch_shared_loss / len(val_loader)
            avg_fw_loss_val = val_epoch_fw_loss / len(val_loader)

            print(f"Validation  Loss: {avg_epoch_loss_val:.5f} | Ortho: {avg_ortho_loss_val:.5f} | Unique: {avg_unique_loss_val:.5f} | Shared: {avg_shared_loss_val:.5f} | Fixed Weight Loss: {avg_fw_loss_val:.5f}")
            
            if avg_fw_loss_val < best_fw_val_loss:
                print(f"New best model found at epoch {_iter + 1} with validation fixed weight loss: {avg_fw_loss_val:.5f} (previous best: {best_fw_val_loss:.5f})")
                best_fw_val_loss = avg_fw_loss_val
                best_model_state = copy.deepcopy(model.state_dict())

            # Log metrics to WandB
            wandb.log({
                "val/loss": avg_epoch_loss_val,
                "val/loss/ortho": avg_ortho_loss_val,
                "val/loss/unique": avg_unique_loss_val,
                "val/loss/shared": avg_shared_loss_val,
                "val/loss/fixed_weight": avg_fw_loss_val,
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
        if (_iter + 1) == epochs and best_model_state is not None:
            checkpoint_name = "final_best_checkpoint.pt"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            os.makedirs(checkpoint_dir, exist_ok=True) # ensure directory exists
            # Create the state dictionary
            checkpoint = {
                'epoch': _iter + 1,
                'model_state_dict': best_model_state
            }
            
            # Save locally
            torch.save(checkpoint, checkpoint_path)

            print(f"Best model checkpoint saved at {checkpoint_path}")
            
            log_model_checkpoint(wandb.run, checkpoint_path, epoch= _iter + 1)

            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            else:
                print("No validation set provided, using final epoch model for evaluation on test set.")
                model.eval()


            # Evaluate linear probe accuracy
            print(f"Evaluating linear and regression probes on train and validation data...")
            train_data_dict = extract_latents_and_labels(model, train_loader, device)
            test_data_dict = extract_latents_and_labels(model, test_loader, device)

            if components is None: # set components only once - same for all epochs
                components = list(train_data_dict['Labels_U'].keys()) + list(train_data_dict['Labels_S'].keys())

            evaluator.set_data(train_data_dict= train_data_dict, val_data_dict= test_data_dict, M= M)
            
            linear_results = evaluator.calculate_linear_probe()
            reg_results = evaluator.calculate_reg_probe()

            metrics_summary = evaluator.mean_metrics(linear_results, reg_results, M= M)
            
            
            print("Evaluation complete!")
            # Log the complete confusion matrix for each epoch (linear)
            wandb.log({"linear_confusion_matrix": wandb.Image(plot_confusion_matrix(linear_results["acc"], components= components, labels= components))})
            

            # Log the pairwise confusion matrices (linear), i.e. M* (M -1)/ 2 matrices for M modalities
            wandb.log({"linear_pairwise_confusion_matrices": wandb.Image(plot_pairwise_confusion_matrices(linear_probe_acc= linear_results["acc"], \
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