import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


SURVIVAL_MIN_EPS = 1e-7


@dataclass
class SurvivalProbeConfig:
    n_bins: int = 4
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    val_fraction: float = 0.2
    patience: int = 10
    min_delta: float = 1e-4
    seed: int = 0
    hidden_dim: int | None = None


class DiscreteTimeSurvivalProbe(nn.Module):
    def __init__(self, input_dim, n_bins, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            self.head = nn.Linear(input_dim, n_bins)
        else:
            self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(hidden_dim, n_bins),
            )

    def forward(self, x):
        return self.head(x)


def make_discrete_time_bins(train_times, train_events, n_bins):
    uncensored_times = train_times[train_events == 1]
    reference_times = uncensored_times if len(uncensored_times) > 0 else train_times
    if len(reference_times) == 0:
        raise ValueError('No valid survival times available to build discrete time bins.')

    quantiles = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)[1:-1]
    if len(quantiles) == 0:
        return np.asarray([], dtype=np.float32)
    return np.unique(np.quantile(reference_times, quantiles).astype(np.float32))


def assign_time_bins(times, bin_edges):
    return np.digitize(times, bin_edges, right=False).astype(np.int64)


def discrete_time_nll(logits, time_bins, events):
    hazards = torch.sigmoid(logits).clamp(min=SURVIVAL_MIN_EPS, max=1.0 - SURVIVAL_MIN_EPS)
    survival = torch.cumprod(1.0 - hazards, dim=1).clamp(min=SURVIVAL_MIN_EPS, max=1.0)

    sample_idx = torch.arange(logits.shape[0], device=logits.device)
    hazard_at_bin = hazards[sample_idx, time_bins]
    survival_at_bin = survival[sample_idx, time_bins]

    survival_before_bin = torch.ones_like(hazard_at_bin)
    has_previous_bin = time_bins > 0
    survival_before_bin[has_previous_bin] = survival[sample_idx[has_previous_bin], time_bins[has_previous_bin] - 1]

    uncensored_loss = -(torch.log(survival_before_bin) + torch.log(hazard_at_bin))
    censored_loss = -torch.log(survival_at_bin)
    return torch.where(events == 1, uncensored_loss, censored_loss).mean()


def compute_risk_from_logits(logits):
    hazards = torch.sigmoid(logits)
    survival = torch.cumprod(1.0 - hazards, dim=1)
    return -survival.sum(dim=1)


def concordance_index(event_times, risk_scores, event_indicators):
    n_samples = len(event_times)
    if n_samples < 2:
        return float('nan')

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
        return float('nan')
    return float((concordant + 0.5 * tied) / comparable)


def subset_survival_arrays(features, survival_data, cancer_type=None):
    mask = survival_data['valid'].astype(bool)
    if cancer_type is not None:
        mask &= survival_data['cancer_type'] == cancer_type

    return {
        'features': np.asarray(features[mask], dtype=np.float32),
        'time': survival_data['time'][mask].astype(np.float32),
        'event': survival_data['event'][mask].astype(np.int64),
        'cancer_type': survival_data['cancer_type'][mask],
    }


def _make_train_val_split(features, times, events, config):
    n_samples = len(features)
    if n_samples < 4:
        return None

    indices = np.arange(n_samples)
    stratify = events if len(np.unique(events)) > 1 and np.min(np.bincount(events)) >= 2 else None
    val_size = max(1, int(round(n_samples * config.val_fraction)))
    if n_samples - val_size < 2:
        val_size = n_samples - 2
    if val_size < 1:
        return None

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_size,
        random_state=config.seed,
        shuffle=True,
        stratify=stratify,
    )
    return {
        'train': (features[train_idx], times[train_idx], events[train_idx]),
        'val': (features[val_idx], times[val_idx], events[val_idx]),
    }


def _standardize(train_x, *others):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    outputs = [(train_x - mean) / std]
    outputs.extend((array - mean) / std for array in others)
    return outputs


def _make_loader(features, time_bins, events, batch_size, shuffle, seed):
    dataset = TensorDataset(
        torch.from_numpy(features.astype(np.float32)),
        torch.from_numpy(time_bins.astype(np.int64)),
        torch.from_numpy(events.astype(np.int64)),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        generator=generator,
    )


def _run_epoch(model, loader, optimizer, device):
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    total_loss = 0.0
    total_samples = 0

    for batch_features, batch_bins, batch_events in loader:
        batch_features = batch_features.to(device)
        batch_bins = batch_bins.to(device)
        batch_events = batch_events.to(device)

        logits = model(batch_features)
        loss = discrete_time_nll(logits, batch_bins, batch_events)

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_size = batch_features.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def fit_and_evaluate_survival_probe(train_features, train_survival, test_features, test_survival, cancer_type=None, config=None):
    config = copy.deepcopy(config) if config is not None else SurvivalProbeConfig()
    train_subset = subset_survival_arrays(train_features, train_survival, cancer_type=cancer_type)
    test_subset = subset_survival_arrays(test_features, test_survival, cancer_type=cancer_type)

    if len(train_subset['features']) < 4 or len(test_subset['features']) < 2:
        return {
            'c_index': float('nan'),
            'test_loss': float('nan'),
            'best_val_loss': float('nan'),
            'best_epoch': 0,
            'epochs_trained': 0,
            'n_train': int(len(train_subset['features'])),
            'n_val': 0,
            'n_test': int(len(test_subset['features'])),
            'n_bins': 0,
        }

    split = _make_train_val_split(
        train_subset['features'],
        train_subset['time'],
        train_subset['event'],
        config,
    )
    if split is None:
        return {
            'c_index': float('nan'),
            'test_loss': float('nan'),
            'best_val_loss': float('nan'),
            'best_epoch': 0,
            'epochs_trained': 0,
            'n_train': int(len(train_subset['features'])),
            'n_val': 0,
            'n_test': int(len(test_subset['features'])),
            'n_bins': 0,
        }

    X_train, time_train, event_train = split['train']
    X_val, time_val, event_val = split['val']

    bin_edges = make_discrete_time_bins(time_train, event_train, config.n_bins)
    n_output_bins = len(bin_edges) + 1

    train_bins = assign_time_bins(time_train, bin_edges)
    val_bins = assign_time_bins(time_val, bin_edges)
    test_bins = assign_time_bins(test_subset['time'], bin_edges)

    X_train, X_val, X_test = _standardize(X_train, X_val, test_subset['features'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DiscreteTimeSurvivalProbe(X_train.shape[1], n_output_bins, hidden_dim=config.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    train_loader = _make_loader(X_train, train_bins, event_train, config.batch_size, True, config.seed)
    val_loader = _make_loader(X_val, val_bins, event_val, config.batch_size, False, config.seed)

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        _run_epoch(model, train_loader, optimizer, device)
        with torch.no_grad():
            val_loss = _run_epoch(model, val_loader, None, device)

        if val_loss < best_val_loss - config.min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(torch.from_numpy(X_test.astype(np.float32)).to(device))
        test_loss = discrete_time_nll(
            test_logits,
            torch.from_numpy(test_bins.astype(np.int64)).to(device),
            torch.from_numpy(test_subset['event'].astype(np.int64)).to(device),
        ).item()
        risk_scores = compute_risk_from_logits(test_logits).detach().cpu().numpy()

    return {
        'c_index': concordance_index(test_subset['time'], risk_scores, test_subset['event']),
        'test_loss': float(test_loss),
        'best_val_loss': float(best_val_loss),
        'best_epoch': int(best_epoch),
        'epochs_trained': int(best_epoch + patience_counter if best_epoch > 0 else 0),
        'n_train': int(len(X_train)),
        'n_val': int(len(X_val)),
        'n_test': int(len(X_test)),
        'n_bins': int(n_output_bins),
    }


def evaluate_feature_survival_analysis(train_features, train_survival, test_features, test_survival, cancer_types=None, config=None):
    train_valid_count = int(train_survival['valid'].astype(bool).sum())
    test_valid_count = int(test_survival['valid'].astype(bool).sum())
    available_cancer_types = sorted(
        set(train_survival['cancer_type'][train_survival['valid'].astype(bool)])
        | set(test_survival['cancer_type'][test_survival['valid'].astype(bool)])
    )
    if not available_cancer_types:
        raise ValueError(
            'No valid survival annotations were found in the provided train/test features. '
            f'Valid train samples: {train_valid_count}, valid test samples: {test_valid_count}.'
        )

    if cancer_types is None:
        selected_cancer_types = [str(cancer_type) for cancer_type in available_cancer_types]
    elif isinstance(cancer_types, str):
        selected_cancer_types = [str(cancer_types)]
    else:
        selected_cancer_types = [str(cancer_type) for cancer_type in cancer_types]
    missing_cancer_types = sorted(set(selected_cancer_types) - set(available_cancer_types))
    if missing_cancer_types:
        raise ValueError(
            f'Requested survival cancer types were not found in the valid survival subset: {missing_cancer_types}. '
            f'Available cancer types: {available_cancer_types}'
        )

    survival_metrics = {}
    for component_name in sorted(train_features.keys()):
        component_metrics = {}
        print(
            f'Evaluating survival probe for component {component_name} with '
            f'{train_valid_count} valid train samples and {test_valid_count} valid test samples '
            f'across cancer types: {selected_cancer_types}'
        )
        for cancer_type in selected_cancer_types:
            component_metrics[str(cancer_type)] = fit_and_evaluate_survival_probe(
                train_features[component_name],
                train_survival,
                test_features[component_name],
                test_survival,
                cancer_type=cancer_type,
                config=config,
            )
        survival_metrics[component_name] = component_metrics
        print(f'Finished evaluating component {component_name}. Metrics by cancer type: {component_metrics}')

    return survival_metrics
