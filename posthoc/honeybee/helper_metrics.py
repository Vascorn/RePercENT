import os
import sys
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

HONEYBEE_MODALITIES = ["clinical_qwen", "pathology_qwen", "wsi", "molecular"]
HONEYBEE_MODALITY_SHORT_NAMES = {
    "clinical_qwen": "clin",
    "pathology_qwen": "path",
    "wsi": "wsi",
    "molecular": "mol",
}



def get_honeybee_modality_short_name(modality_name):
    return HONEYBEE_MODALITY_SHORT_NAMES.get(modality_name, modality_name)


def format_honeybee_component_name(component_type, source_modality, target_modality):
    source_short = get_honeybee_modality_short_name(source_modality)
    target_short = get_honeybee_modality_short_name(target_modality)
    return f"{component_type}_{source_short}_{target_short}"


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
    X, X_aug, X_cross_masks, X_aug_cross_masks = _prepare_honeybee_batch(batch, device, modality_order=modality_order)
    outputs = model(X, mask=X_cross_masks)
    outputs_aug = model(X_aug, mask=X_aug_cross_masks)
    loss, logs = disen_loss(outputs, outputs_aug)
    return outputs, loss, logs


def test_fwd_only(batch, model, device, modality_order=None):
    X, _, X_cross_masks, _ = _prepare_honeybee_batch(batch, device, modality_order=modality_order)
    return model(X, mask=X_cross_masks)


def train_loop(batch, model, disen_loss, optimizer, device, modality_order=None):
    X, X_aug, X_cross_masks, X_aug_cross_masks = _prepare_honeybee_batch(batch, device, modality_order=modality_order)
    outputs = model(X, mask=X_cross_masks)
    outputs_aug = model(X_aug, mask=X_aug_cross_masks)
    loss, logs = disen_loss(outputs, outputs_aug)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, logs


def _collect_component_features(loader, model, device, modality_order=None):
    modality_order = modality_order or HONEYBEE_MODALITIES
    component_features = defaultdict(list)
    labels = []

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            outputs = test_fwd_only(batch, model, device, modality_order=modality_order)
            labels.extend([str(label) for label in batch["cancer_type"]])

            U = outputs["U"].detach().cpu()
            S_view = outputs["S_view"].detach().cpu()
            _, n_modalities, _, _ = U.shape

            for i in range(n_modalities):
                for j in range(n_modalities):
                    if i == j:
                        continue
                    source_modality = modality_order[i]
                    target_modality = modality_order[j]
                    component_features[format_honeybee_component_name("U", source_modality, target_modality)].append(U[:, i, j, :])
                    component_features[format_honeybee_component_name("S", source_modality, target_modality)].append(S_view[:, i, j, :])
                    component_features[format_honeybee_component_name("D", source_modality, target_modality)].append(torch.cat((U[:, i, j, :], S_view[:, i, j, :]), dim=-1))

                component_features[format_honeybee_component_name("S", modality_order[i], "all")].append(
                    torch.cat([S_view[:, i, k, :] for k in range(n_modalities) if k != i], dim=-1)
                )
                component_features[format_honeybee_component_name("D", modality_order[i], "all")].append(
                    torch.cat([
                        torch.cat((U[:, i, k, :], S_view[:, i, k, :]), dim=-1)
                        for k in range(n_modalities) if k != i
                    ], dim=-1)
                )

    feature_arrays = {name: torch.cat(chunks, dim=0).numpy() for name, chunks in component_features.items()}


    return feature_arrays, np.asarray(labels)


def extract_acc_per_cancer_type(y_pred, y_true, label_encoder):
    acc_per_cancer_type = {}
    for idx, cancer_type in enumerate(label_encoder.classes_):
        class_mask = y_true == idx
        if not np.any(class_mask):
            acc_per_cancer_type[cancer_type] = float("nan")
            continue
        acc_per_cancer_type[cancer_type] = float(np.mean(y_pred[class_mask] == y_true[class_mask]))
    return acc_per_cancer_type


def evaluate_model_cancer_type(train_loader, test_loader, model, device, modality_order=None):
    modality_order = modality_order or HONEYBEE_MODALITIES
    train_features, train_labels = _collect_component_features(train_loader, model, device, modality_order=modality_order)
    test_features, test_labels = _collect_component_features(test_loader, model, device, modality_order=modality_order)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    y_test = label_encoder.transform(test_labels)

    probe_acc = {}
    for component_name in sorted(train_features.keys()):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(train_features[component_name], y_train)
        y_pred = clf.predict(test_features[component_name])
        probe_acc[component_name] = {
            "overall": float(accuracy_score(y_test, y_pred)),
            "per_cancer_type": extract_acc_per_cancer_type(y_pred, y_test, label_encoder),
        }

    return probe_acc

