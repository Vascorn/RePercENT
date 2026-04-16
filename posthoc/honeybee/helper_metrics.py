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

from posthoc.honeybee.survival_probe import SurvivalProbeConfig, evaluate_feature_survival_analysis


HONEYBEE_MODALITIES = ["clinical_qwen", "pathology_qwen", "wsi", "molecular"]
HONEYBEE_MODALITY_SHORT_NAMES = {
    "clinical_qwen": "clin",
    "pathology_qwen": "path",
    "wsi": "wsi",
    "molecular": "mol",
}


SURVIVAL_PROBE_CONFIG = SurvivalProbeConfig(
    n_bins=4,
    epochs=100,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=32,
    val_fraction=0.2,
    patience=10,
    min_delta=1e-4,
    seed=0,
)


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


def _safe_float(value):
    if isinstance(value, torch.Tensor):
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return np.nan
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"", "unknown", "nan", "na", "none", "not reported", "--"}:
            return np.nan
        try:
            return float(cleaned)
        except ValueError:
            return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _is_observed_event(vital_status):
    if vital_status is None:
        return None
    status = str(vital_status).strip().lower()
    if status in {"dead", "deceased", "1", "true"}:
        return 1
    if status in {"alive", "living", "0", "false"}:
        return 0
    return None


def _extract_survival_targets_from_batch(batch):
    batch_size = len(batch["cancer_type"])
    survival = {
        "patient_id": [],
        "event": [],
        "time": [],
        "valid": [],
        "cancer_type": [],
    }

    patient_ids = batch.get("patient_id", [None] * batch_size)
    vital_statuses = batch.get("vital_status", [None] * batch_size)
    days_to_death = batch.get("days_to_death", [None] * batch_size)
    days_to_last_follow_up = batch.get("days_to_last_follow_up", [None] * batch_size)
    cancer_types = batch["cancer_type"]

    for idx in range(batch_size):
        event = _is_observed_event(vital_statuses[idx])
        death_days = _safe_float(days_to_death[idx])
        follow_up_days = _safe_float(days_to_last_follow_up[idx])

        if event == 1:
            time = death_days
        elif event == 0:
            time = follow_up_days
        else:
            time = np.nan

        valid = event is not None and np.isfinite(time) and time >= 0.0
        survival["patient_id"].append(str(patient_ids[idx]) if patient_ids[idx] is not None else "unknown")
        survival["event"].append(int(event) if event is not None else -1)
        survival["time"].append(float(time) if np.isfinite(time) else np.nan)
        survival["valid"].append(bool(valid))
        survival["cancer_type"].append(str(cancer_types[idx]))

    return survival


def _collect_component_features(loader, model, device, modality_order=None):
    modality_order = modality_order or HONEYBEE_MODALITIES
    component_features = defaultdict(list)
    labels = []
    survival_data = defaultdict(list)

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            outputs = test_fwd_only(batch, model, device, modality_order=modality_order)
            labels.extend([str(label) for label in batch["cancer_type"]])

            batch_survival = _extract_survival_targets_from_batch(batch)
            for key, values in batch_survival.items():
                survival_data[key].extend(values)

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
    survival_arrays = {key: np.asarray(values) for key, values in survival_data.items()}
    return feature_arrays, np.asarray(labels), survival_arrays


def extract_acc_per_cancer_type(y_pred, y_true, label_encoder, component_name):
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
    train_features, train_labels, _ = _collect_component_features(train_loader, model, device, modality_order=modality_order)
    test_features, test_labels, _ = _collect_component_features(test_loader, model, device, modality_order=modality_order)

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
            "per_cancer_type": extract_acc_per_cancer_type(y_pred, y_test, label_encoder, component_name),
        }

    return probe_acc


def evaluate_model_survival_analysis(train_loader, test_loader, model, device, modality_order=None, cancer_types=None, probe_config=None):
    modality_order = modality_order or HONEYBEE_MODALITIES
    train_features, _, train_survival = _collect_component_features(train_loader, model, device, modality_order=modality_order)
    test_features, _, test_survival = _collect_component_features(test_loader, model, device, modality_order=modality_order)

    return evaluate_feature_survival_analysis(
        train_features,
        train_survival,
        test_features,
        test_survival,
        cancer_types=cancer_types,
        config=probe_config or SURVIVAL_PROBE_CONFIG,
    )
