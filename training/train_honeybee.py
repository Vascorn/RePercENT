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
import re
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from src.utils.helpers import plot_pairwise_confusion_matrices


HONEYBEE_MODALITIES = ["clinical_qwen", "pathology_qwen", "wsi", "molecular"]


def _prepare_honeybee_batch(batch, device, modality_order=None):
    modality_order = modality_order or HONEYBEE_MODALITIES

    X, X_aug = [], []
    X_cross_masks, X_aug_cross_masks = [], []

    for modality in modality_order:
        embeddings, aug_embeddings, pad_mask, has_data = batch[modality]

        if isinstance(has_data, torch.Tensor):
            if not bool(has_data.all()):
                raise ValueError(f"Batch contains missing data for modality {modality}. Use aligned patients or a custom collate path.")
        elif not bool(has_data):
            raise ValueError(f"Batch contains missing data for modality {modality}.")

        X.append(embeddings.to(device))
        X_aug.append(aug_embeddings.to(device))

        mask = pad_mask.bool().to(device) if pad_mask is not None else None
        X_cross_masks.append(mask)
        X_aug_cross_masks.append(mask)

    return X, X_aug, X_cross_masks, X_aug_cross_masks


def test_loop(batch, model, disen_loss, device, modality_order=None):
    X, X_aug, X_cross_masks, X_aug_cross_masks = _prepare_honeybee_batch(
        batch,
        device,
        modality_order=modality_order,
    )

    outputs = model(X, mask=X_cross_masks)
    outputs_aug = model(X_aug, mask=X_aug_cross_masks)
    loss, logs = disen_loss(outputs, outputs_aug)

    return outputs, loss, logs


def test_fwd_only(batch, model, device, modality_order=None):
    X, _, X_cross_masks, _ = _prepare_honeybee_batch(
        batch,
        device,
        modality_order=modality_order,
    )
    return model(X, mask=X_cross_masks)


def train_loop(batch, model, disen_loss, optimizer, device, modality_order=None):
    X, X_aug, X_cross_masks, X_aug_cross_masks = _prepare_honeybee_batch(
        batch,
        device,
        modality_order=modality_order,
    )

    outputs = model(X, mask=X_cross_masks)
    outputs_aug = model(X_aug, mask=X_aug_cross_masks)
    loss, logs = disen_loss(outputs, outputs_aug)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, logs

def _collect_component_features(loader, model, device, modality_order=None):
    component_features = defaultdict(list)
    labels = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = test_fwd_only(batch, model, device, modality_order=modality_order)
            labels.extend([str(label) for label in batch["cancer_type"]])

            U = outputs["U"].detach().cpu()
            S_view = outputs["S_view"].detach().cpu()
            _, M, _, _ = U.shape

            for i in range(M):
                for j in range(M):
                    if i == j:
                        continue
                    component_features[f"U_{i + 1}{j + 1}"].append(U[:, i, j, :])
                    component_features[f"S_{i + 1}{j + 1}"].append(S_view[:, i, j, :])
    
    component_features = {
        name: torch.cat(chunks, dim=0).numpy()
        for name, chunks in component_features.items()
    }
    return component_features, np.asarray(labels)


def _evaluate_cancer_type_linear_probes(train_loader, test_loader, model, device, modality_order=None):
    train_features, train_labels = _collect_component_features(
        train_loader,
        model,
        device,
        modality_order=modality_order,
    )
    test_features, test_labels = _collect_component_features(
        test_loader,
        model,
        device,
        modality_order=modality_order,
    )

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    y_test = label_encoder.transform(test_labels)

    probe_acc = {}
    for component_name in sorted(train_features.keys()):
        clf = LogisticRegression(max_iter=3000)
        clf.fit(train_features[component_name], y_train)
        y_pred = clf.predict(test_features[component_name])
        probe_acc[component_name] = float(accuracy_score(y_test, y_pred))

    return probe_acc


def _component_display_name(component_key, modality_order):
    match = re.match(r"^([US])_(\d)(\d)$", component_key)
    if not match:
        return component_key

    prefix, i_str, j_str = match.groups()
    i, j = int(i_str), int(j_str)
    if i <= 0 or j <= 0 or i > len(modality_order) or j > len(modality_order):
        return component_key

    return f"{prefix}_{modality_order[i - 1]}_{modality_order[j - 1]}"


def _build_pairwise_probe_plot(probe_acc, modality_order, include_reverse_shared=False):
    if not probe_acc:
        return None

    modality_order = modality_order or HONEYBEE_MODALITIES
    sorted_components = sorted(probe_acc.keys())
    sorted_components_lower = [key.lower() for key in sorted_components]

    # Build a sparse square matrix-like dict expected by plot_pairwise_confusion_matrices.
    # Each component contributes only its own measured probe accuracy on the diagonal.
    # Pairwise blocks are 3x3 by default and can be switched to 4x4 via include_reverse_shared.
    n_components = len(sorted_components_lower)
    idx_by_key = {key: idx for idx, key in enumerate(sorted_components_lower)}
    linear_probe_acc = {}
    for original_key, lowered_key in zip(sorted_components, sorted_components_lower):
        row = np.full(n_components, np.nan, dtype=float)
        row[idx_by_key[lowered_key]] = float(probe_acc[original_key]) * 100.0
        linear_probe_acc[lowered_key] = row

    key_display_map = {
        key.lower(): _component_display_name(key, modality_order)
        for key in sorted_components
    }
    pairs = [(i, j) for i in range(len(modality_order)) for j in range(i + 1, len(modality_order))]
    
    return plot_pairwise_confusion_matrices(
        linear_probe_acc=linear_probe_acc,
        M=len(modality_order),
        components=sorted_components_lower,
        pairs=pairs,
        key_display_map=key_display_map,
        modality_names=modality_order,
        include_reverse_shared=include_reverse_shared,
        vmin= 0.0,
        vmax= 100.0,
    )



def train(train_loader, test_loader, model, optimizer, disen_loss, epochs, device, val_loader= None, checkpoint_dir="./checkpoints", modality_order= None, include_reverse_shared_pairwise= False, evaluate_final_model=False):
    """
    Full training loop for RePercENT on the Honeybee multimodal TCGA dataset.

    The Honeybee dataset already yields original and augmented embeddings per modality,
    so training and evaluation are loss-based and use the modality tuples directly.
    """
    torch.cuda.empty_cache()
    os.makedirs(checkpoint_dir, exist_ok=True)

    wandb.watch(model, log="gradients")
    print(f'Number of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    overall_best_val_loss = float('inf')
    overall_best_state_dict = None
    overall_best_epoch = 0

    for _iter in range(epochs):
        epoch_loss = 0.0
        epoch_ortho_loss = 0.0
        epoch_unique_loss = 0.0
        epoch_shared_loss = 0.0
        epoch_fw_loss = 0.0

        model.train()
        print(f"----- Epoch: {_iter + 1} / {epochs} -----")
        for batch_idx, batch in enumerate(train_loader):
            loss, logs = train_loop(
                batch,
                model,
                disen_loss,
                optimizer,
                device,
                modality_order=modality_order,
            )
            epoch_loss += loss.item()
            epoch_ortho_loss += logs['ortho']
            epoch_unique_loss += logs['unique']
            epoch_shared_loss += logs['shared']
            epoch_fw_loss += logs['fw_loss']

        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_ortho_loss = epoch_ortho_loss / len(train_loader)
        avg_unique_loss = epoch_unique_loss / len(train_loader)
        avg_shared_loss = epoch_shared_loss / len(train_loader)
        avg_fw_loss = epoch_fw_loss / len(train_loader)
        print(f"Training  Loss: {avg_epoch_loss:.5f} | Ortho: {avg_ortho_loss:.5f} | Unique: {avg_unique_loss:.5f} | Shared: {avg_shared_loss:.5f} | fw: {avg_fw_loss:.5f} | Lmd: {disen_loss.lmd:.6f}, alpha: {disen_loss.alpha:.6f}")

        wandb.log({
            "train/loss": avg_epoch_loss,
            "train/loss/ortho": avg_ortho_loss,
            "train/loss/unique": avg_unique_loss,
            "train/loss/shared": avg_shared_loss,
            "train/loss/fw": avg_fw_loss,
        }, step=_iter + 1)

        if val_loader is not None:
            val_epoch_loss = 0.0
            val_epoch_ortho_loss = 0.0
            val_epoch_unique_loss = 0.0
            val_epoch_shared_loss = 0.0
            val_epoch_fw_loss = 0.0

            model.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    _, val_loss, val_logs = test_loop(
                        batch,
                        model,
                        disen_loss,
                        device,
                        modality_order=modality_order,
                    )
                    val_epoch_loss += val_loss.item()
                    val_epoch_ortho_loss += val_logs['ortho']
                    val_epoch_unique_loss += val_logs['unique']
                    val_epoch_shared_loss += val_logs['shared']
                    val_epoch_fw_loss += val_logs['fw_loss']

            avg_epoch_loss_val = val_epoch_loss / len(val_loader)
            avg_ortho_loss_val = val_epoch_ortho_loss / len(val_loader)
            avg_unique_loss_val = val_epoch_unique_loss / len(val_loader)
            avg_shared_loss_val = val_epoch_shared_loss / len(val_loader)
            avg_fw_loss_val = val_epoch_fw_loss / len(val_loader)

            print(f"Validation  Loss: {avg_epoch_loss_val:.5f} | Ortho: {avg_ortho_loss_val:.5f} | Unique: {avg_unique_loss_val:.5f} | Shared: {avg_shared_loss_val:.5f} | fw: {avg_fw_loss_val:.5f}")

            wandb.log({
                "val/loss": avg_epoch_loss_val,
                "val/loss/ortho": avg_ortho_loss_val,
                "val/loss/unique": avg_unique_loss_val,
                "val/loss/shared": avg_shared_loss_val,
                "val/loss/fw": avg_fw_loss_val,
            }, step=_iter + 1)

            if avg_fw_loss_val < overall_best_val_loss:
                overall_best_val_loss = avg_fw_loss_val
                overall_best_state_dict = copy.deepcopy(model.state_dict())
                overall_best_epoch = _iter + 1
                print(f"New best model found at epoch {overall_best_epoch} with validation loss {overall_best_val_loss:.5f}")

        if (_iter + 1) % 10 == 0 or (_iter + 1) == epochs:
            checkpoint_name = f"checkpoint_epoch_{_iter + 1}.pt" if (_iter + 1) // 10 != (epochs // 10) else "final_checkpoint.pt"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint = {
                'epoch': _iter + 1,
                'model_state_dict': copy.deepcopy(model.state_dict())
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")
            log_model_checkpoint(wandb.run, checkpoint_path, epoch=_iter + 1)

    if epochs > 0 and val_loader is not None:
        print(f"Best model found at epoch {overall_best_epoch} with validation loss {overall_best_val_loss:.5f}")
        checkpoint_path = os.path.join(checkpoint_dir, "best_model_overall.pt")
        checkpoint = {
            'epoch': overall_best_epoch,
            'model_state_dict': overall_best_state_dict,
        }
        torch.save(checkpoint, checkpoint_path)
        log_model_checkpoint(wandb.run, checkpoint_path, epoch=overall_best_epoch, extra_meta={"best_overall": True})

    print("Training complete!")
    if evaluate_final_model:

        if overall_best_state_dict is not None:
            print(f"Loading best model from epoch {overall_best_epoch} with validation loss {overall_best_val_loss:.5f} for final testing on test set...")
            model.load_state_dict(overall_best_state_dict)

        test_epoch_loss = 0.0
        test_epoch_ortho_loss = 0.0
        test_epoch_unique_loss = 0.0
        test_epoch_shared_loss = 0.0
        test_epoch_fw_loss = 0.0

        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                _, test_loss, test_logs = test_loop(
                    batch,
                    model,
                    disen_loss,
                    device,
                    modality_order=modality_order,
                )
                test_epoch_loss += test_loss.item()
                test_epoch_ortho_loss += test_logs['ortho']
                test_epoch_unique_loss += test_logs['unique']
                test_epoch_shared_loss += test_logs['shared']
                test_epoch_fw_loss += test_logs['fw_loss']

        avg_test_loss = test_epoch_loss / len(test_loader)
        avg_test_ortho = test_epoch_ortho_loss / len(test_loader)
        avg_test_unique = test_epoch_unique_loss / len(test_loader)
        avg_test_shared = test_epoch_shared_loss / len(test_loader)
        avg_test_fw = test_epoch_fw_loss / len(test_loader)

        probe_acc = _evaluate_cancer_type_linear_probes(
            train_loader,
            test_loader,
            model,
            device,
            modality_order=modality_order,
        )

        metrics_summary = {
            "test/loss": avg_test_loss,
            "test/loss/ortho": avg_test_ortho,
            "test/loss/unique": avg_test_unique,
            "test/loss/shared": avg_test_shared,
            "test/loss/fw": avg_test_fw,
        }
        for component_name, acc in probe_acc.items():
            metrics_summary[f"probe/cancer_type/{component_name}/acc"] = acc
        if val_loader is not None:
            metrics_summary["best_val_epoch"] = overall_best_epoch
            metrics_summary["best_val_loss"] = overall_best_val_loss

        table = wandb.Table(columns=["metric", "value"])
        for k, v in metrics_summary.items():
            table.add_data(k, float(v))
        wandb.log({"final_metrics": table})
        wandb.log({
            f"probe/cancer_type/{component_name}/acc": acc
            for component_name, acc in probe_acc.items()
        })

        pairwise_fig = _build_pairwise_probe_plot(
            probe_acc,
            modality_order,
            include_reverse_shared=include_reverse_shared_pairwise,
        )
        if pairwise_fig is not None:
            print("Logging pairwise linear probe accuracy plot")
            wandb.log({"pairwise_confusion_matrices": wandb.Image(pairwise_fig)})
            # save figure locally as well for easier access
            pairwise_fig_path = os.path.join(checkpoint_dir, "pairwise_probe_accuracy.pdf")
            pairwise_fig.savefig(pairwise_fig_path, dpi= 150, bbox_inches='tight')
            print(f"Pairwise probe accuracy plot saved at {pairwise_fig_path}")

        print("Cancer-type linear probe accuracy per component:")
        for component_name, acc in probe_acc.items():
            print(f"  {component_name}: {acc:.4f}")
            
        return metrics_summary
    
    return None
