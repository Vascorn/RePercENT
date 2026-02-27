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
import copy

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


def calc_batch_correct(outputs, distractors, device, comp_mod= 1):

    # NOTE; the models' forward pass is order sensitive
    # 1: text, 2: definitions, 0: is the imaging modality
    
    
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

def test_loop(x, x_aug, model, disen_loss, device, M= 3):
    images, texts, text_mask = x["images"], x["texts"], x["pad_masks"]
    
    images_aug, texts_aug, text_mask_aug = x_aug["images"], x_aug["texts"], x_aug["pad_masks"]
    X = [images.to(device), texts.to(device)]
    X_cross_masks = [None, text_mask.bool().to(device)] 
    
    X_aug = [images_aug.to(device), texts_aug.to(device)]
    X_aug_cross_masks = [None, text_mask_aug.bool().to(device)]


    if M == 3:
        defs, defs_mask = x["definitions"], x["definitions_mask"]
        defs_aug, defs_mask_aug = x_aug["definitions"], x_aug["definitions_mask"]
    
        X.append(defs.to(device))
        X_cross_masks.append(defs_mask.bool().to(device))
        X_aug.append(defs_aug.to(device))
        X_aug_cross_masks.append(defs_mask_aug.bool().to(device))

    # Forward pass through RePercENT
    outputs = model(X, mask = X_cross_masks)
    outputs_aug = model(X_aug, mask = X_aug_cross_masks)
    
    # Compute disentanglement loss
    loss, logs = disen_loss(outputs, outputs_aug)

    return outputs, loss, logs

def test_fwd_only(x, model, device, M= 3):
    images, texts, text_mask = x["images"], x["texts"], x["pad_masks"]
    
    X = [images.to(device), texts.to(device)]
    X_cross_masks = [None, text_mask.bool().to(device)] 

    if M == 3:
        defs, defs_mask = x["definitions"], x["definitions_mask"]
    
        X.append(defs.to(device))
        X_cross_masks.append(defs_mask.bool().to(device))

    # Forward pass through RePercENT
    outputs = model(X, mask = X_cross_masks)
    
    return outputs

def train_loop(x, x_aug, model, disen_loss, optimizer, device, M= 3):
    images, texts, text_mask = x["images"], x["texts"], x["pad_masks"]
    
    images_aug, texts_aug, text_mask_aug = x_aug["images"], x_aug["texts"], x_aug["pad_masks"]
    X = [images.to(device), texts.to(device)]
    X_cross_masks = [None, text_mask.bool().to(device)] 
    
    X_aug = [images_aug.to(device), texts_aug.to(device)]
    X_aug_cross_masks = [None, text_mask_aug.bool().to(device)]


    if M == 3:
        defs, defs_mask = x["definitions"], x["definitions_mask"]
        defs_aug, defs_mask_aug = x_aug["definitions"], x_aug["definitions_mask"]
    
        X.append(defs.to(device))
        X_cross_masks.append(defs_mask.bool().to(device))
        X_aug.append(defs_aug.to(device))
        X_aug_cross_masks.append(defs_mask_aug.bool().to(device))

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

def train(train_loader, test_loader, model, optimizer, disen_loss, epochs, device, val_loader=None, checkpoint_dir="./checkpoints", comp_mod = 1):
    """
    Full training loop for RePercENT model with evaluation on test set
    Args:
        train_loader: DataLoader for training dataset
        test_loader: DataLoader for test dataset. Used for final evaluation after training.
        model: RePercENT model (This training function also supports JointOpt model variants)
        optimizer: Optimizer for RePercENT model
        disen_loss: Disentanglement loss function
        epochs: Number of training epochs
        device: Device to run the training on (CPU/GPU)
        val_loader: DataLoader for validation dataset. If provided, the model will be evaluated on this set at the end of each epoch and the best model checkpoint will be saved based on validation loss.
        checkpoint_dir: Directory to save model checkpoints
        comp_mod: Integer indicating which modality to compare for the final evaluation. For M=2 (images + captions), set comp_mod = 1 to 
        compare images to captions. For M=3 (images + captions + definitions), set comp_mod = 1 to compare images to captions, and comp_mod = 2 to compare images to definitions.
    Returns:
        A dictionary summarizing final metrics on the test set.
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

    # If validation loader is provided, we will save the best model based on validation loss and optionally test it on the test set at the end.
    overall_best_val_loss = float('inf')
    overall_best_state_dict = None
    overall_best_epoch = 0
    # evaluator = ProbeEvaluator(linear_probe= linear_probe, regression_probe= regression_probe)
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
        for batch_idx, out in enumerate(train_loader):

            x = out['x']
            x_aug = out['x_aug']

            temp_b = x['images'].shape[0]
            
            loss, logs = train_loop(x, x_aug, model, disen_loss, optimizer, device, M= M)

            # # Track losses
            epoch_loss += loss.item()
            epoch_ortho_loss += logs['ortho']
            epoch_unique_loss += logs['unique']
            epoch_shared_loss += logs['shared']
            epoch_fw_loss += logs['fw_loss']
        # Epoch statistics
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_ortho_loss = epoch_ortho_loss / len(train_loader)
        avg_unique_loss = epoch_unique_loss / len(train_loader)
        avg_shared_loss = epoch_shared_loss / len(train_loader)
        avg_fw_loss = epoch_fw_loss / len(train_loader)
        print(f"Training  Loss: {avg_epoch_loss:.5f} | Ortho: {avg_ortho_loss:.5f} | Unique: {avg_unique_loss:.5f} | Shared: {avg_shared_loss:.5f} | fw: {avg_fw_loss:.5f} | Lmd: {disen_loss.lmd:.6f}, alpha: {disen_loss.alpha:.6f}")
        # Log metrics to WandB
        wandb.log({
            "train/loss": avg_epoch_loss,
            "train/loss/ortho": avg_ortho_loss,
            "train/loss/unique": avg_unique_loss,
            "train/loss/shared": avg_shared_loss,
            "train/loss/fw": avg_fw_loss
        }, step= _iter + 1)

        # Calculate loss & accuracy on test set
        if val_loader is not None:
            val_epoch_loss = 0.0
            val_epoch_ortho_loss = 0.0
            val_epoch_unique_loss = 0.0
            val_epoch_shared_loss = 0.0
            val_epoch_fw_loss = 0.0
            
            model.eval()
            with torch.no_grad():
                for batch_idx, out in enumerate(val_loader):
                    x = out['x']
                    x_aug = out['x_aug']
                    temp_b = x['images'].shape[0]
                    
                    outputs, val_loss, val_logs = test_loop(x, x_aug, model, disen_loss, device, M= M)
                    val_epoch_loss += val_loss.item()   
                    val_epoch_ortho_loss += val_logs['ortho']
                    val_epoch_unique_loss += val_logs['unique']
                    val_epoch_shared_loss += val_logs['shared']
                    val_epoch_fw_loss += val_logs['fw_loss']
            # Epoch statistics
            avg_epoch_loss_val = val_epoch_loss / len(val_loader)
            avg_ortho_loss_val = val_epoch_ortho_loss / len(val_loader)
            avg_unique_loss_val = val_epoch_unique_loss / len(val_loader)
            avg_shared_loss_val = val_epoch_shared_loss / len(val_loader)
            avg_fw_loss_val = val_epoch_fw_loss / len(val_loader)
        
            
            print(f"Validation  Loss: {avg_epoch_loss_val:.5f} | Ortho: {avg_ortho_loss_val:.5f} | Unique: {avg_unique_loss_val:.5f} | Shared: {avg_shared_loss_val:.5f} | fw: {avg_fw_loss_val:.5f}")

            # Log metrics to WandB
            wandb.log({
                "val/loss": avg_epoch_loss_val,
                "val/loss/ortho": avg_ortho_loss_val,
                "val/loss/unique": avg_unique_loss_val,
                "val/loss/shared": avg_shared_loss_val,
                "val/loss/fw": avg_fw_loss_val
            }, step= _iter + 1)
        

            # if avg_fw_loss_val < overall_best_val_loss:
            #     overall_best_val_loss = avg_fw_loss_val
            #     overall_best_state_dict = copy.deepcopy(model.state_dict())
            #     overall_best_epoch = _iter + 1
            #     print(f"New best model found at epoch {overall_best_epoch} with validation loss {overall_best_val_loss:.5f}")

        # Save model checkpoint every 10 epochs and at the end
        if (_iter + 1) % 10 == 0 or (_iter + 1) == epochs:
            checkpoint_name = f"checkpoint_epoch_{_iter + 1}.pt" if (_iter + 1) // 10 != (epochs // 10) else f"final_checkpoint.pt"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            os.makedirs(checkpoint_dir, exist_ok=True) # ensure directory exists
            # Create the state dictionary
            checkpoint = {
                'epoch': _iter + 1,
                'model_state_dict': copy.deepcopy(model.state_dict())
            }
            
            # Save locally
            torch.save(checkpoint, checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")
            
            log_model_checkpoint(wandb.run, checkpoint_path, epoch= _iter + 1)
    
    if _iter + 1 == epochs and val_loader is not None:
        print(f"Best model found at epoch {overall_best_epoch} with validation loss {overall_best_val_loss:.5f}")
        checkpoint_name = f"best_model_overall.pt"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        # Create the state dictionary        
        checkpoint = {
            'epoch': overall_best_epoch,
            'model_state_dict': overall_best_state_dict
        }
        torch.save(checkpoint, checkpoint_path)
        log_model_checkpoint(wandb.run, checkpoint_path, epoch= overall_best_epoch, extra_meta={"best_overall": True})

    # Calculate accuracy on test set
    overall_correct = 0
    overall_total = 0

    # track per type
    type_correct = defaultdict(int)
    type_total   = defaultdict(int)
    
    if comp_mod == 2 and M == 2:
        raise ValueError("comp_mod is set to 2 but M is 2, which means there are only 2 modalities (images + captions). Please set comp_mod to 1 to compare images to captions, or set M to 3 if you want to compare images to definitions.")
    
    # If the validation set is used, we will test the best model on the test set. Otherwise, we will test the final epoch model.
    if overall_best_state_dict is not None:
        print(f"Loading best model from epoch {overall_best_epoch} with validation loss {overall_best_val_loss:.5f} for final testing on test set...")
        model.load_state_dict(overall_best_state_dict)

    model.eval()
    with torch.no_grad():
        for batch_idx, out in enumerate(test_loader):
            x = out['x']
            x_aug = out['x_aug']
            temp_b = x['images'].shape[0]
            
            outputs = test_fwd_only(x, model, device, M= M)
        
            distractors = out['distractors'].to(device)

            B, N, S, D = distractors.shape
            distr_flat = rearrange(distractors, 'b n s d -> (b n) s d')
            if hasattr(model, "disenEncoders"): # This is the RePercENT case
                out_distractors_flat = model.disenEncoders[0](distr_flat)[:, 1, :]

            elif hasattr(model, "sharedEncoders"): # This is the general JointOpt case
                out_distractors_flat = model.encode_modality(model.sharedEncoders[f"S_1{comp_mod + 1}"], \
                                model.sharedProjh[f"S_1{comp_mod + 1}"],distr_flat, None)

            out_distractors = rearrange(out_distractors_flat, '(b n) ... -> b n ...', b=B, n=N)


            correct = calc_batch_correct(outputs, out_distractors, device, comp_mod= comp_mod)
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
        if val_loader is not None:
            metrics_summary["best_val_epoch"] = overall_best_epoch
            metrics_summary["best_val_loss"] = overall_best_val_loss

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
            epoch_loss += loss.item()
            epoch_ortho_loss += logs['ortho']
            epoch_unique_loss += logs['unique']
            epoch_shared_loss += logs['shared']
            
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
                val_epoch_loss += val_loss.item()
                val_epoch_ortho_loss += val_logs['ortho']
                val_epoch_unique_loss += val_logs['unique']
                val_epoch_shared_loss += val_logs['shared']
        
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

        print(f"Training  Loss: {avg_epoch_loss:.5f} | Ortho: {avg_ortho_loss:.5f} | Unique: {avg_unique_loss:.5f} | Shared: {avg_shared_loss:.5f} | Lmd: {disen_loss.lmd:.6f}, alpha: {disen_loss.alpha:.6f}")
        print(f"Testing  Loss: {avg_epoch_loss_val:.5f} | Ortho: {avg_ortho_loss_val:.5f} | Unique: {avg_unique_loss_val:.5f} | Shared: {avg_shared_loss_val:.5f}")

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
