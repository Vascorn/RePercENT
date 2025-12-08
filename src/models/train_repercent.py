import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Literal, List
from src.DisentangledSSL.models import ProbabilisticEncoder 
from src.DisentangledSSL.losses import SupConLoss, ortho_loss
from src.utils.supervised import linearprobe
from src.DisentangledSSL.dataset import augment_data

import wandb

def init_wandb(project: str = "RePercENT", name: str | None = None, config: dict | None = None, tags: list | None = None, entity: str | None = None):
    """
    Initialize a wandb run if none exists. Returns the active run.
    """
    if wandb.run is None:
        run = wandb.init(project=project, name=name, config=config, tags=tags, entity=entity)
    else:
        run = wandb.run
        # optionally update config if provided
        if config:
            wandb.config.update(config, allow_val_change=True)
    return run

def watch_model(model, log: str = "all", log_freq: int = 100):
    """
    Attach model to wandb to track gradients/parameters.
    Safe to call even if wandb is not initialized yet.
    """
    try:
        wandb.watch(model, log=log, log_freq=log_freq)
    except Exception:
        # if wandb not initialized or watching fails, ignore silently
        pass

def wrap_dissen_loss(dissen_loss, prefix: str = "train"):
    """
    Wrap the provided dissen_loss so that every call logs metrics to wandb.
    Expected dissen_loss signature: (out, out_aug) -> (loss_tensor, logs_dict)
    Logs are prefixed with `prefix/`.
    """
    def wrapped(out, out_aug):
        loss, logs = dissen_loss(out, out_aug)
        # normalize logs to dict
        logs = {} if logs is None else dict(logs)
        # scalarize loss
        try:
            loss_val = loss.item()
        except Exception:
            try:
                loss_val = float(loss)
            except Exception:
                loss_val = None

        metrics = {}
        if loss_val is not None:
            metrics[f"{prefix}/loss"] = loss_val

        for k, v in logs.items():
            try:
                metrics[f"{prefix}/{k}"] = v.item() if hasattr(v, "item") else float(v)
            except Exception:
                metrics[f"{prefix}/{k}"] = v

        # log metrics to wandb (batch-level). Commit=False allows batching if needed.
        try:
            wandb.log(metrics, commit=False)
        except Exception:
            # safe fallback if wandb not initialized or fails
            pass

        return loss, logs

    return wrapped


def train_loop(model, train_loader, optimizer, dissen_loss, noise_scale=0.01, drop_scale=10):
    epoch_loss = 0.0
    epoch_logs = []
    for i_batch, data_batch in enumerate(train_loader):
        model.train()
        x1 = data_batch[0].float().cuda()
        x2 = data_batch[1].float().cuda()
        x1 = augment_data(x1, noise_scale, drop_scale)
        x2 = augment_data(x2, noise_scale, drop_scale)
        x1_aug = augment_data(x1, noise_scale, drop_scale)
        x2_aug = augment_data(x2, noise_scale, drop_scale)
        out = model(x1, x2)
        out_aug = model(x1_aug, x2_aug)
        loss_train, logs_train = dissen_loss(out, out_aug)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        epoch_loss += loss_train.item()
        epoch_logs.append(logs_train)
    epoch_loss /= len(train_loader)
    return epoch_loss, epoch_logs


def train_model(model, train_loader, test_loader, dissen_loss, optimizer, num_epoch=50, noise_scale=0.01, drop_scale=10):
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epoch, eta_min=0, last_epoch=-1)
        train_loss_logs = {"loss": [], "logs": {}}
        test_loss_logs = {"loss": [], "logs": {}}
        for _iter in range(num_epoch):
            print("----- Epoch: " + str(num_epoch) + " -----")
            epoch_loss = 0.0
            model.train()
            epoch_loss, epoch_logs = train_loop(model, train_loader, optimizer, dissen_loss, noise_scale, drop_scale)
            train_loss_logs["loss"].append(epoch_loss)
            train_loss_logs["logs"][_iter] = epoch_logs
            print(f"Train Loss: {epoch_loss:.4f}")


            model.eval()
            with torch.no_grad():
                epoch_test_loss = 0.0
                for i_batch, data_batch in enumerate(test_loader):
                    x1 = data_batch[0].float().cuda()
                    x2 = data_batch[1].float().cuda()
                    x1 = augment_data(x1, noise_scale, drop_scale)
                    x2 = augment_data(x2, noise_scale, drop_scale)
                    x1_aug = augment_data(x1, noise_scale, drop_scale)
                    x2_aug = augment_data(x2, noise_scale, drop_scale)
                    out = model(x1, x2)
                    out_aug = model(x1_aug, x2_aug)
                    loss_test, logs_test = dissen_loss(out, out_aug)
                    epoch_test_loss += loss_test.item()
                epoch_test_loss /= len(test_loader)
                test_loss_logs["loss"].append(epoch_test_loss)
                test_loss_logs["logs"][_iter] = logs_test
                print(f"Test Loss: {epoch_test_loss:.4f}")
                #TODO: linear probe evaluation every n epochs
            lr_scheduler.step()
        return train_loss_logs, test_loss_logs