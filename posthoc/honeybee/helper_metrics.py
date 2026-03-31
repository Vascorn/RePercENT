import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch.nn as nn
import torch
import torch.nn.functional as F
import typing
from typing import Literal, List
from src.models import repercent, jointopt
import numpy as np
from einops import rearrange
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader


HONEYBEE_MODALITIES = ["clinical_qwen", "pathology_qwen", "wsi", "molecular"]
HONEYBEE_MODALITY_SHORT_NAMES = {
    "clinical_qwen": "clin",
    "pathology_qwen": "path",
    "wsi": "wsi",
    "molecular": "mol",
}

SURVIVAL_N_BINS = 4
SURVIVAL_EPOCHS = 100
SURVIVAL_LR = 1e-3
SURVIVAL_WEIGHT_DECAY = 1e-4
SURVIVAL_BATCH_SIZE = 32
SURVIVAL_MIN_EPS = 1e-7


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
    with torch.no_grad():
        for batch in loader:
            outputs = test_fwd_only(batch, model, device, modality_order=modality_order)
            labels.extend([str(label) for label in batch["cancer_type"]])

            batch_survival = _extract_survival_targets_from_batch(batch)
            for key, values in batch_survival.items():
                survival_data[key].extend(values)

            U = outputs["U"].detach().cpu()
            S_view = outputs["S_view"].detach().cpu()
            _, M, _, _ = U.shape

            for i in range(M):
                for j in range(M):
                    if i == j:
                        continue
                    source_modality = modality_order[i]
                    target_modality = modality_order[j]
                    component_features[
                        format_honeybee_component_name("U", source_modality, target_modality)
                    ].append(U[:, i, j, :])
                    component_features[
                        format_honeybee_component_name("S", source_modality, target_modality)
                    ].append(S_view[:, i, j, :])
                    component_features[
                        format_honeybee_component_name("D", source_modality, target_modality)
                    ].append(torch.cat((U[:, i, j, :], S_view[:, i, j, :]), dim=-1))

                component_features[
                    format_honeybee_component_name("S", modality_order[i], "all")
                ].append(torch.cat([S_view[:, i, k, :] for k in range(M) if k != i], dim=-1))
                component_features[
                    format_honeybee_component_name("D", modality_order[i], "all")
                ].append(torch.cat([
                    torch.cat((U[:, i, k, :], S_view[:, i, k, :]), dim=-1)
                    for k in range(M) if k != i
                ], dim=-1))

    component_features = {
        name: torch.cat(chunks, dim=0).numpy()
        for name, chunks in component_features.items()
    }
    survival_data = {key: np.asarray(values) for key, values in survival_data.items()}
    return component_features, np.asarray(labels), survival_data


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
    train_features, train_labels, _ = _collect_component_features(
        train_loader,
        model,
        device,
        modality_order=modality_order,
    )
    test_features, test_labels, _ = _collect_component_features(
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
        clf = LogisticRegression(max_iter=2000)
        clf.fit(train_features[component_name], y_train)
        y_pred = clf.predict(test_features[component_name])
        probe_acc[component_name] = {
            "overall": float(accuracy_score(y_test, y_pred)),
            "per_cancer_type": extract_acc_per_cancer_type(y_pred, y_test, label_encoder, component_name),
        }

    return probe_acc


class _LinearDiscreteTimeSurvivalProbe(nn.Module):
    def __init__(self, input_dim, n_bins):
        super().__init__()
        self.head = nn.Linear(input_dim, n_bins)

    def forward(self, x):
        return self.head(x)


def _make_discrete_time_bins(train_times, train_events, n_bins):
    uncensored_times = train_times[train_events == 1]
    reference_times = uncensored_times if len(uncensored_times) > 0 else train_times
    if len(reference_times) == 0:
        raise ValueError("No valid survival times available to build discrete time bins.")

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    if len(quantiles) == 0:
        return np.asarray([], dtype=np.float32)

    bin_edges = np.quantile(reference_times, quantiles)
    return np.unique(np.asarray(bin_edges, dtype=np.float32))


def _assign_time_bins(times, bin_edges):
    return np.digitize(times, bin_edges, right=False).astype(np.int64)


def _discrete_time_nll(logits, time_bins, events):
    hazards = torch.sigmoid(logits).clamp(min=SURVIVAL_MIN_EPS, max=1.0 - SURVIVAL_MIN_EPS)
    survival = torch.cumprod(1.0 - hazards, dim=1).clamp(min=SURVIVAL_MIN_EPS, max=1.0)

    sample_indices = torch.arange(logits.shape[0], device=logits.device)
    hazard_at_bin = hazards[sample_indices, time_bins]
    survival_at_bin = survival[sample_indices, time_bins]

    survival_before_bin = torch.ones_like(hazard_at_bin)
    has_previous_bin = time_bins > 0
    survival_before_bin[has_previous_bin] = survival[sample_indices[has_previous_bin], time_bins[has_previous_bin] - 1]

    uncensored_loss = -(torch.log(survival_before_bin) + torch.log(hazard_at_bin))
    censored_loss = -torch.log(survival_at_bin)
    return torch.where(events == 1, uncensored_loss, censored_loss).mean()


def _compute_risk_from_logits(logits):
    hazards = torch.sigmoid(logits)
    survival = torch.cumprod(1.0 - hazards, dim=1)
    return -survival.sum(dim=1)


def _concordance_index(event_times, risk_scores, event_indicators):
    n_samples = len(event_times)
    if n_samples < 2:
        return float("nan")

    concordant = 0.0
    tied = 0.0
    comparable = 0.0

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            time_i, time_j = event_times[i], event_times[j]
            event_i, event_j = event_indicators[i], event_indicators[j]
            risk_i, risk_j = risk_scores[i], risk_scores[j]

            if event_i == 1 and time_i < time_j:
                comparable += 1.0
                if risk_i > risk_j:
                    concordant += 1.0
                elif risk_i == risk_j:
                    tied += 1.0
            elif event_j == 1 and time_j < time_i:
                comparable += 1.0
                if risk_j > risk_i:
                    concordant += 1.0
                elif risk_i == risk_j:
                    tied += 1.0

    if comparable == 0.0:
        return float("nan")
    return float((concordant + 0.5 * tied) / comparable)


def _subset_survival_arrays(features, survival_data, cancer_type=None):
    mask = survival_data["valid"].astype(bool)
    if cancer_type is not None:
        mask &= survival_data["cancer_type"] == cancer_type

    subset = {
        "features": features[mask],
        "time": survival_data["time"][mask].astype(np.float32),
        "event": survival_data["event"][mask].astype(np.int64),
        "cancer_type": survival_data["cancer_type"][mask],
    }
    return subset


def _prepare_survival_probe_data(features, survival_data, bin_edges, cancer_type=None):
    subset = _subset_survival_arrays(features, survival_data, cancer_type=cancer_type)
    valid_time_bins = _assign_time_bins(subset["time"], bin_edges)
    return subset["features"], subset["time"], subset["event"], valid_time_bins


def _fit_and_evaluate_survival_probe(train_features, train_survival, test_features, test_survival, n_bins=SURVIVAL_N_BINS, cancer_type=None):
    train_subset = _subset_survival_arrays(train_features, train_survival, cancer_type=cancer_type)
    test_subset = _subset_survival_arrays(test_features, test_survival, cancer_type=cancer_type)

    if len(train_subset["features"]) < 2 or len(test_subset["features"]) < 2:
        return {
            "c_index": float("nan"),
            "test_loss": float("nan"),
            "n_train": int(len(train_subset["features"])),
            "n_test": int(len(test_subset["features"])),
            "n_bins": 0,
        }

    train_times = train_subset["time"]
    train_events = train_subset["event"]
    bin_edges = _make_discrete_time_bins(train_times, train_events, n_bins)
    n_output_bins = len(bin_edges) + 1

    X_train, train_times, train_events, train_time_bins = _prepare_survival_probe_data(
        train_features,
        train_survival,
        bin_edges,
        cancer_type=cancer_type,
    )
    X_test, test_times, test_events, test_time_bins = _prepare_survival_probe_data(
        test_features,
        test_survival,
        bin_edges,
        cancer_type=cancer_type,
    )

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    train_mean = X_train.mean(axis=0, keepdims=True)
    train_std = X_train.std(axis=0, keepdims=True)
    train_std[train_std < 1e-6] = 1.0
    X_train = (X_train - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std

    probe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    survival_model = _LinearDiscreteTimeSurvivalProbe(X_train.shape[1], n_output_bins).to(probe_device)
    optimizer = torch.optim.AdamW(survival_model.parameters(), lr=SURVIVAL_LR, weight_decay=SURVIVAL_WEIGHT_DECAY)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(train_time_bins),
        torch.from_numpy(train_events),
    )
    train_loader = DataLoader(train_dataset, batch_size=min(SURVIVAL_BATCH_SIZE, len(train_dataset)), shuffle=True)

    survival_model.train()
    for _ in range(SURVIVAL_EPOCHS):
        for batch_features, batch_bins, batch_events in train_loader:
            batch_features = batch_features.to(probe_device)
            batch_bins = batch_bins.to(probe_device)
            batch_events = batch_events.to(probe_device)

            logits = survival_model(batch_features)
            loss = _discrete_time_nll(logits, batch_bins, batch_events)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    survival_model.eval()
    with torch.no_grad():
        test_logits = survival_model(torch.from_numpy(X_test).to(probe_device))
        test_loss = _discrete_time_nll(
            test_logits,
            torch.from_numpy(test_time_bins).to(probe_device),
            torch.from_numpy(test_events).to(probe_device),
        ).item()
        risk_scores = _compute_risk_from_logits(test_logits).detach().cpu().numpy()

    return {
        "c_index": _concordance_index(test_times, risk_scores, test_events),
        "test_loss": float(test_loss),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_bins": int(n_output_bins),
    }


def evaluate_model_survival_analysis(train_loader, test_loader, model, device, modality_order=None, cancer_types=None):
    modality_order = modality_order or HONEYBEE_MODALITIES
    train_features, _, train_survival = _collect_component_features(
        train_loader,
        model,
        device,
        modality_order=modality_order,
    )
    test_features, _, test_survival = _collect_component_features(
        test_loader,
        model,
        device,
        modality_order=modality_order,
    )

    train_valid_count = int(train_survival["valid"].astype(bool).sum())
    test_valid_count = int(test_survival["valid"].astype(bool).sum())
    available_cancer_types = sorted(
        set(train_survival["cancer_type"][train_survival["valid"].astype(bool)])
        | set(test_survival["cancer_type"][test_survival["valid"].astype(bool)])
    )
    if not available_cancer_types:
        raise ValueError(
            "No valid survival annotations were found in the provided train/test loaders. "
            f"Valid train samples: {train_valid_count}, valid test samples: {test_valid_count}. "
            "The current dataset appears to expose 'unknown' for vital_status and survival times, so survival analysis cannot be computed until those fields are populated upstream."
        )

    selected_cancer_types = [str(cancer_type) for cancer_type in (cancer_types or available_cancer_types)]

    missing_cancer_types = sorted(set(selected_cancer_types) - set(available_cancer_types))
    if missing_cancer_types:
        raise ValueError(
            f"Requested survival cancer types were not found in the valid survival subset: {missing_cancer_types}. "
            f"Available cancer types: {available_cancer_types}"
        )

    survival_metrics = {}
    for component_name in sorted(train_features.keys()):
        component_metrics = {}

        for cancer_type in selected_cancer_types:
            component_metrics[str(cancer_type)] = _fit_and_evaluate_survival_probe(
                train_features[component_name],
                train_survival,
                test_features[component_name],
                test_survival,
                n_bins=SURVIVAL_N_BINS,
                cancer_type=cancer_type,
            )

        survival_metrics[component_name] = component_metrics

    return survival_metrics
