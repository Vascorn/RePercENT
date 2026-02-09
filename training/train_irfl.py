import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import torch.nn as nn
from typing import Literal, List
import wandb
from training.log_data import log_model_checkpoint
import numpy as np
import math
from einops import rearrange
import torch.nn.functional as F
from collections import defaultdict


def calc_metrics_summary(overall_correct, overall_total, type_correct, type_total):
    """
    Create a summary of metrics including overall accuracy and per figurative type accuracy.
    Args:
        overall_correct: Total number of correct predictions across all samples.
        overall_total: Total number of samples.
        type_correct: Dictionary with figurative type as keys and number of correct predictions as values.
        type_total: Dictionary with figurative type as keys and total number of samples as values.
    Returns:
        A dictionary summarizing overall and per-type metrics.
    """

    metrics_summary = {
        "overall/accuracy": overall_correct / max(1, overall_total),
        "overall/correct": overall_correct,
        "overall/total": overall_total,
    }

    for t in ["idiom", "metaphor", "simile"]:
        tot = type_total.get(t, 0)
        cor = type_correct.get(t, 0)
        metrics_summary[t + "/accuracy"] = (cor / tot) if tot > 0 else float("nan")
        metrics_summary[t + "/correct"] = cor
        metrics_summary[t + "/total"] = tot     

    return metrics_summary


def calc_batch_correct(outputs, distractors, device, comp2caption= True):
    # 1: text, 2: definitions, 0: is the imaging modality
    # NOTE; RePercENT's forward pass is order sensitive
    comp_mod = 1 if comp2caption else 2 
    # shared_text: [B, D]
    shared_text = outputs['S_view'][:, comp_mod, 0]
    shared_text = F.normalize(shared_text, dim=-1)

    # shared_image_answers: [B, D]
    shared_image_answers = outputs['S_view'][:, 0, comp_mod]
    shared_image_answers = F.normalize(shared_image_answers, dim=-1)

    # shared_image_distractors: [B, K, D]
    shared_image_distractors = distractors
    shared_image_distractors = F.normalize(shared_image_distractors, dim=-1)

    # answer_sim: [B]
    answer_sim = (shared_text * shared_image_answers).sum(dim=-1)


    # distractor_sims: [B, K]
    # einsum computes dot(shared_text[b], shared_image_distractors[b, k]) for all b,k
    distractor_sims = torch.einsum('bd,bkd->bk', shared_text, shared_image_distractors)

    # max_distractor_sim: [B]
    max_distractor_sim = distractor_sims.max(dim=1).values
    
    # correct: [B] bool
    correct = answer_sim > max_distractor_sim
    
    return correct

def test_loop(x, x_aug, model, disen_loss, device):
    images, texts, text_mask, defs, defs_mask = x["images"], x["texts"], x["pad_masks"], x["definitions"], x["definitions_mask"]
    images_aug, texts_aug, text_mask_aug, defs_aug, defs_mask_aug = x_aug["images"], x_aug["texts"], x_aug["pad_masks"], x_aug["definitions"], x_aug["definitions_mask"]
    
    X = [images.to(device), texts.to(device), defs.to(device)]
    X_cross_masks = [None, text_mask.bool().to(device), defs_mask.bool().to(device)] 
    
    X_aug = [images_aug.to(device), texts_aug.to(device), defs_aug.to(device)]
    X_aug_cross_masks = [None, text_mask_aug.bool().to(device), defs_mask_aug.bool().to(device)]

    # Forward pass through RePercENT
    outputs = model(X, mask = X_cross_masks)
    outputs_aug = model(X_aug, mask = X_aug_cross_masks)
    
    # Compute disentanglement loss
    loss, logs = disen_loss(outputs, outputs_aug)

    return outputs, loss, logs


def train_loop(x, x_aug, model, disen_loss, optimizer, device):
    images, texts, text_mask, defs, defs_mask = x["images"], x["texts"], x["pad_masks"], x["definitions"], x["definitions_mask"]
    images_aug, texts_aug, text_mask_aug, defs_aug, defs_mask_aug = x_aug["images"], x_aug["texts"], x_aug["pad_masks"], x_aug["definitions"], x_aug["definitions_mask"]
    
    X = [images.to(device), texts.to(device), defs.to(device)]
    X_cross_masks = [None, text_mask.bool().to(device), defs_mask.bool().to(device)] 
    
    X_aug = [images_aug.to(device), texts_aug.to(device), defs_aug.to(device)]
    X_aug_cross_masks = [None, text_mask_aug.bool().to(device), defs_mask_aug.bool().to(device)]

    # Forward pass through RePercENT
    outputs = model(X, mask = X_cross_masks)
    outputs_aug = model(X_aug, mask = X_aug_cross_masks)
    
    # Compute disentanglement loss
    loss, logs = disen_loss(outputs, outputs_aug)
    
    # Backward pass for RePercENT
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, logs

def train(train_loader, val_loader, test_loader, model, optimizer, disen_loss, epochs, device, checkpoint_dir="./checkpoints"):
    """
    Full training loop for RePercENT model with evaluation on test set
    Args:
        train_loader: DataLoader for training dataset
        val_loader: DataLoader for validation dataset
        test_loader: DataLoader for test dataset. Used for final evaluation after training.
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
    
    # evaluator = ProbeEvaluator(linear_probe= linear_probe, regression_probe= regression_probe)
    for _iter in range(epochs):
        # Initialize epoch loss trackers
        epoch_loss = 0.0
        epoch_ortho_loss = 0.0
        epoch_unique_loss = 0.0
        epoch_shared_loss = 0.0
        
        model.train()
        print(f"----- Epoch: {_iter + 1} / {epochs} -----")
        # Training phase
        for batch_idx, out in enumerate(train_loader):

            x = out['x']
            x_aug = out['x_aug']

            temp_b = x['images'].shape[0]
            
            loss, logs = train_loop(x, x_aug, model, disen_loss, optimizer, device)

            # # Track losses
            epoch_loss += loss.item() / temp_b
            epoch_ortho_loss += logs['ortho'] / temp_b
            epoch_unique_loss += logs['unique'] / temp_b
            epoch_shared_loss += logs['shared'] / temp_b
            
        # Epoch statistics
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_ortho_loss = epoch_ortho_loss / len(train_loader)
        avg_unique_loss = epoch_unique_loss / len(train_loader)
        avg_shared_loss = epoch_shared_loss / len(train_loader)

        # Calculate loss & accuracy on test set
        val_epoch_loss = 0.0
        val_epoch_ortho_loss = 0.0
        val_epoch_unique_loss = 0.0
        val_epoch_shared_loss = 0.0

        model.eval()
        with torch.no_grad():
            for batch_idx, out in enumerate(val_loader):
                x = out['x']
                x_aug = out['x_aug']
                temp_b = x['images'].shape[0]
                
                outputs, val_loss, val_logs = test_loop(x, x_aug, model, disen_loss, device)
                val_epoch_loss += val_loss.item() / temp_b
                val_epoch_ortho_loss += val_logs['ortho'] / temp_b
                val_epoch_unique_loss += val_logs['unique'] / temp_b
                val_epoch_shared_loss += val_logs['shared'] / temp_b
        
        # Epoch statistics
        avg_epoch_loss_val = val_epoch_loss / len(val_loader)
        avg_ortho_loss_val = val_epoch_ortho_loss / len(val_loader)
        avg_unique_loss_val = val_epoch_unique_loss / len(val_loader)
        avg_shared_loss_val = val_epoch_shared_loss / len(val_loader)
        

        print(f"Training  Loss(x 100): {avg_epoch_loss* 100:.5f} | Ortho (x 100): {avg_ortho_loss* 100:.5f} | Unique (x 100): {avg_unique_loss* 100:.5f} | Shared (x 100): {avg_shared_loss* 100:.5f} | Lmd: {disen_loss.lmd:.6f}, alpha: {disen_loss.alpha:.6f}")
        print(f"Testing  Loss(x 100): {avg_epoch_loss_val* 100:.5f} | Ortho (x 100): {avg_ortho_loss_val* 100:.5f} | Unique (x 100): {avg_unique_loss_val* 100:.5f} | Shared (x 100): {avg_shared_loss_val* 100:.5f}")


        
        # Log metrics to WandB
        wandb.log({
            "train/loss": avg_epoch_loss,
            "train/loss/ortho": avg_ortho_loss,
            "train/loss/unique": avg_unique_loss,
            "train/loss/shared": avg_shared_loss,
            "val/loss": avg_epoch_loss_val,
            "val/loss/ortho": avg_ortho_loss_val,
            "val/loss/unique": avg_unique_loss_val,
            "val/loss/shared": avg_shared_loss_val
        }, step= _iter + 1)
        
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
    
    
    # Calculate accuracy on test set
    overall_correct = 0
    overall_total = 0

    # track per type
    type_correct = defaultdict(int)
    type_total   = defaultdict(int)

    model.eval()
    with torch.no_grad():
        for batch_idx, out in enumerate(test_loader):
            x = out['x']
            x_aug = out['x_aug']
            temp_b = x['images'].shape[0]
            
            outputs, val_loss, val_logs = test_loop(x, x_aug, model, disen_loss, device)
        
            distractors = out['distractors'].to(device)

            B, N, S, D = distractors.shape
            distr_flat = rearrange(distractors, 'b n s d -> (b n) s d')
            out_distractors_flat = model.disenEncoders[0](distr_flat)[:, 1, :]

            out_distractors = rearrange(out_distractors_flat, '(b n) ... -> b n ...', b=B, n=N)


            correct = calc_batch_correct(outputs, out_distractors, device)
            # ---- overall ----
            overall_correct += correct.sum().item()
            overall_total += correct.numel()

            # ---- by figurative type ----
            # out["figurative_type"] is usually a list of strings length B
            ftypes = out["figurative_type"]
            
            for t, c in zip(ftypes, correct.tolist()):
                # normalize spelling just in case: "metaphore" vs "metaphor"
                t = str(t).lower().strip()
                type_total[t] += 1
                type_correct[t] += int(c)
            
        accuracy = overall_correct / max(1, overall_total)

    # Save final metrics:
    if (_iter + 1) == epochs:
        metrics_summary = calc_metrics_summary(overall_correct, overall_total, type_correct, type_total)

        # Log final metrics table
        table = wandb.Table(columns=["metric", "value"])
        for k, v in metrics_summary.items():
            table.add_data(k, float(v))
        wandb.log({"final_metrics": table})
    
    print("Training complete!")
    return metrics_summary



def train_sweep(train_loader, val_loader, model, optimizer, disen_loss, epochs, device):
    """
    Full training loop for RePercENT model with evaluation on validation set
    Args:
        train_loader: DataLoader for training dataset
        val_loader: DataLoader for validation dataset
        model: RePercENT model (This training function also supports the JointOpt model)
        optimizer: Optimizer for RePercENT model
        disen_loss: Disentanglement loss function
        epochs: Number of training epochs
        device: Device to run the training on (CPU/GPU)
    """
    # clear memory
    torch.cuda.empty_cache()
    
    
    # Watch model with WandB
    print(f'Number of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    M = disen_loss.M # number of modalities
    
    best_epoch = 0
    best_val_loss = float('inf')
    best_val_loss_unique = float('inf')
    best_val_loss_shared = float('inf')
    best_val_loss_ortho = float('inf')


    # evaluator = ProbeEvaluator(linear_probe= linear_probe, regression_probe= regression_probe)
    for _iter in range(epochs):
        # Initialize epoch loss trackers
        epoch_loss = 0.0
        epoch_ortho_loss = 0.0
        epoch_unique_loss = 0.0
        epoch_shared_loss = 0.0
        
        model.train()
        print(f"----- Epoch: {_iter + 1} / {epochs} -----")
        # Training phase
        for batch_idx, out in enumerate(train_loader):

            x = out['x']
            x_aug = out['x_aug']

            temp_b = x['images'].shape[0]
            
            loss, logs = train_loop(x, x_aug, model, disen_loss, optimizer, device)

            # # Track losses
            epoch_loss += loss.item() / temp_b
            epoch_ortho_loss += logs['ortho'] / temp_b
            epoch_unique_loss += logs['unique'] / temp_b
            epoch_shared_loss += logs['shared'] / temp_b
            
        # Epoch statistics
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_ortho_loss = epoch_ortho_loss / len(train_loader)
        avg_unique_loss = epoch_unique_loss / len(train_loader)
        avg_shared_loss = epoch_shared_loss / len(train_loader)

        # Calculate loss & accuracy on test set
        val_epoch_loss = 0.0
        val_epoch_ortho_loss = 0.0
        val_epoch_unique_loss = 0.0
        val_epoch_shared_loss = 0.0

        model.eval()
        with torch.no_grad():
            for batch_idx, out in enumerate(val_loader):
                x = out['x']
                x_aug = out['x_aug']
                temp_b = x['images'].shape[0]
                
                outputs, val_loss, val_logs = test_loop(x, x_aug, model, disen_loss, device)
                val_epoch_loss += val_loss.item() / temp_b
                val_epoch_ortho_loss += val_logs['ortho'] / temp_b
                val_epoch_unique_loss += val_logs['unique'] / temp_b
                val_epoch_shared_loss += val_logs['shared'] / temp_b
        
        # Epoch statistics
        avg_epoch_loss_val = val_epoch_loss / len(val_loader)
        avg_ortho_loss_val = val_epoch_ortho_loss / len(val_loader)
        avg_unique_loss_val = val_epoch_unique_loss / len(val_loader)
        avg_shared_loss_val = val_epoch_shared_loss / len(val_loader)
        
        if avg_epoch_loss_val < best_val_loss:
            best_val_loss = avg_epoch_loss_val
            best_epoch = _iter + 1

            # NOTE: We count as best losses the ones at best epoch, even if individually they are not the best.
            best_val_loss_ortho = avg_ortho_loss_val
            best_val_loss_unique = avg_unique_loss_val
            best_val_loss_shared = avg_shared_loss_val

        print(f"Training  Loss(x 100): {avg_epoch_loss* 100:.5f} | Ortho (x 100): {avg_ortho_loss* 100:.5f} | Unique (x 100): {avg_unique_loss* 100:.5f} | Shared (x 100): {avg_shared_loss* 100:.5f} | Lmd: {disen_loss.lmd:.6f}, alpha: {disen_loss.alpha:.6f}")
        print(f"Testing  Loss(x 100): {avg_epoch_loss_val* 100:.5f} | Ortho (x 100): {avg_ortho_loss_val* 100:.5f} | Unique (x 100): {avg_unique_loss_val* 100:.5f} | Shared (x 100): {avg_shared_loss_val* 100:.5f}")

        # Log metrics to WandB
        wandb.log({
            "train/loss": avg_epoch_loss,
            "train/loss/ortho": avg_ortho_loss,
            "train/loss/unique": avg_unique_loss,
            "train/loss/shared": avg_shared_loss,
            "val/loss": avg_epoch_loss_val,
            "val/loss/ortho": avg_ortho_loss_val,
            "val/loss/unique": avg_unique_loss_val,
            "val/loss/shared": avg_shared_loss_val
        }, step= _iter + 1)

    final_metrics = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_loss_ortho": best_val_loss_ortho,
        "best_val_loss_unique": best_val_loss_unique,
        "best_val_loss_shared": best_val_loss_shared
    }
    return final_metrics
