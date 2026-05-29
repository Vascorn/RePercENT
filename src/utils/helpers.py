import os, sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List
import torch.nn as nn
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score, recall_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
import re
from typing import Any, Callable, Dict, Optional, Tuple
from dataclasses import dataclass, field
import random
from itertools import combinations
import yaml


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    

def set_seed(seed: int):
    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch (CPU & GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For CUDA >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def plot_confusion_matrix(linear_probe_acc, labels: List= ['labels_1', 'labels_2', 'labels_s'], components: List= ['u_12', 'u_21', 's']):

    fig, axes = plt.subplots(1, 1, figsize=(12, 10))
    

    # Extract the arrays from the dictionary in the specified order
    arrays_to_stack = [linear_probe_acc[key] for key in components]

    # Stack the arrays vertically to create the 4x3 matrix
    result_matrix = np.vstack(arrays_to_stack)
    cm = result_matrix
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=components, ax=axes, vmin=50, vmax=100)
    axes.set_title('Linear Probe Accuracy for Components')
    axes.set_xlabel('Labels')
    axes.set_ylabel('Components')

    plt.tight_layout()
    plt.show()
    return fig


def plot_pairwise_confusion_matrices(
    linear_probe_acc,
    M,
    components: List = ['u_12', 'u_21', 's'],
    pairs: List = None,
    key_display_map: Dict[str, str] = None,
    modality_names: List[str] = None,
    include_reverse_shared: bool = False,
    x_label_rotation_threshold: int = 10,
    x_label_rotation: int = 45,
    vmin: float = 50.0,
    vmax: float = 100.0
):
    """
    Plot M*(M-1)/2 confusion matrices, one per modality pair.

    By default (include_reverse_shared=False), each pair is 3x3: [u_ij, u_ji, s_ij].
    If include_reverse_shared=True, each pair is 4x4: [u_ij, u_ji, s_ij, s_ji].
    """

    if components is None:
        components = list(linear_probe_acc.keys())

    # reconstruct component & label keys (same set)
    label_keys = components.copy()
    comp_keys = components.copy()
    comp_idx = {k: idx for idx, k in enumerate(comp_keys)}

    if pairs is None:
        pairs = list(combinations(range(M), 2))
    x_shape = M if M % 2 else M // 2
    y_shape = M - 1 if (M - 1) % 2 else M // 2
    
    x_shape, y_shape = (y_shape, x_shape) if x_shape > y_shape else (x_shape, y_shape)
    fig, axes = plt.subplots(x_shape, y_shape, figsize=(5 * y_shape, 6 * x_shape))
    axes = np.atleast_1d(axes).ravel()

    key_display_map = key_display_map or {}

    for pair_id, (i, j) in enumerate(pairs):
        
        if modality_names is not None and len(modality_names) >= max(i + 1, j + 1):
            pair_name = f"{modality_names[i]} vs {modality_names[j]}"
        else:
            pair_name = f"{i+1} vs {j+1}"
        
        # columns: 3x3 default or 4x4 with reverse shared component
        col_keys = [f"u_{i+1}{j+1}", f"u_{j+1}{i+1}", f"s_{i+1}{j+1}"]
        if include_reverse_shared:
            col_keys.append(f"s_{j+1}{i+1}")

        # rows: same pairwise labels
        row_keys = col_keys
        submat = np.full((len(row_keys), len(col_keys)), np.nan, dtype=float)

        for r_idx, row_key in enumerate(row_keys):
            if row_key not in linear_probe_acc:
                continue
            row_values = np.asarray(linear_probe_acc[row_key], dtype=float)
            for c_idx, col_key in enumerate(col_keys):
                col_index = comp_idx.get(col_key)
                if col_index is None or col_index >= row_values.shape[0]:
                    continue
                submat[r_idx, c_idx] = row_values[col_index]


        ax = axes[pair_id]
        
        display_cols = [key_display_map.get(k, k) for k in col_keys]
        display_rows = [key_display_map.get(k, k) for k in row_keys]
        rotate_x_labels = any(len(str(label)) > x_label_rotation_threshold for label in display_cols)

        sns.heatmap(
            submat,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=display_cols,
            yticklabels=display_rows,
            cbar=True,
            vmin=vmin,
            vmax=vmax,
            mask=~np.isfinite(submat),
            ax=ax
        )
        ax.set_title(f"Linear Probe – Pairwise Confusion ({pair_name})")
        ax.set_xlabel("Components")
        ax.set_ylabel("Labels")
        if rotate_x_labels:
            plt.setp(ax.get_xticklabels(), rotation=x_label_rotation, ha="right")
        else:
            plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        plt.tight_layout()
    plt.show()
    return fig



def linear_probe(train_data, train_labels, test_data, test_labels):
    # Train logistic regression
    clf = LogisticRegression(max_iter= 10000)
    clf.fit(train_data, train_labels)
    
    # Predict and compute accuracy
    labels_pred = clf.predict(test_data)
    acc = accuracy_score(test_labels, labels_pred) * 100  # Convert to percentage
    mcc = matthews_corrcoef(test_labels, labels_pred)
    f1 = f1_score(test_labels, labels_pred, average='weighted')
    recall = recall_score(test_labels, labels_pred, average='weighted')
    return {"acc": acc, "mcc": mcc, "f1": f1, "recall": recall, "predicted": labels_pred, "true_labels": test_labels}

def non_linear_probe(train_data, train_labels, test_data, test_labels, 
                     hidden_dims=(64, 64), lr=1e-3, num_epochs=200, 
                     batch_size=256, early_stopping_patience=20, device=None):
    """
    Train an MLP classifier probe on the learned representations.
    
    Args:
        train_data: Training features (N, D)
        train_labels: Training labels (N,)
        test_data: Test features (N, D)
        test_labels: Test labels (N,)
        hidden_dims: Tuple of hidden layer dimensions
        lr: Learning rate
        num_epochs: Maximum number of training epochs
        batch_size: Batch size for training
        early_stopping_patience: Stop if val loss doesn't improve for this many epochs
        device: Device to run on (defaults to cuda if available)
    
    Returns:
        Dictionary with acc, mcc, f1, recall, predicted, true_labels
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_dim = train_data.shape[1]
    output_dim = len(np.unique(train_labels))

    # Build MLP
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend([
            nn.Linear(prev_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        ])
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    
    model = nn.Sequential(*layers).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Convert data to PyTorch tensors
    train_data_tensor = torch.FloatTensor(train_data).to(device)
    train_labels_tensor = torch.LongTensor(train_labels).to(device)
    test_data_tensor = torch.FloatTensor(test_data).to(device)
    test_labels_tensor = torch.LongTensor(test_labels).to(device)

    # Create DataLoader for batched training
    train_dataset = torch.utils.data.TensorDataset(train_data_tensor, train_labels_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop with early stopping
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_data_tensor)
        _, predicted = torch.max(test_outputs.data, 1)
        total = test_labels_tensor.size(0)
        correct = (predicted == test_labels_tensor).sum().item()
        acc = (correct / total) * 100  # Convert to percentage
        
        pred_cpu = predicted.cpu().numpy()
        labels_cpu = test_labels_tensor.cpu().numpy()
        
        mcc = matthews_corrcoef(labels_cpu, pred_cpu)
        f1 = f1_score(labels_cpu, pred_cpu, average='weighted')
        recall = recall_score(labels_cpu, pred_cpu, average='weighted')

    return {"acc": acc, "mcc": mcc, "f1": f1, "recall": recall, "predicted": pred_cpu, "true_labels": labels_cpu}


def regression_probe(train_data, train_labels, test_data, test_labels):

    # Train linear regression
    reg = LinearRegression()
    reg.fit(train_data, train_labels)
    
    # Predict and compute MSE
    labels_pred = reg.predict(test_data)
    mse = mean_squared_error(test_labels, labels_pred)
    mae = mean_absolute_error(test_labels, labels_pred)
    r2 = reg.score(test_data, test_labels)

    return {"mae": mae, "mse": mse, "r2": r2, "predicted": labels_pred, "true_labels": test_labels}

# extract all the train-data and labels to have them ready for linear probing

def extract_latents_and_labels(model, loader, device):
    # Initialize tensors to store all the latents & labels
    M = len(loader.dataset[0][0])  # number of modalities
    U_chunks = [[[] for _ in range(M)] for _ in range(M)]
    S_chunks = [[[] for _ in range(M)] for _ in range(M)]
    X_in = [[] for _ in range(M)]

    model.eval()
    with torch.no_grad():
        for batch_idx, (X, labels_u, labels_s, t_u, t_s) in enumerate(loader):
            temp_b = X[0].shape[0]  # batch size
            dim_shape = X[0].shape[-1] # dimension of original Z1 or Z2 
            X = [X[m].to(device) for m in range(len(X))]
            
            outputs = model(X, mask = [None for _ in range(len(X))])

            for m in range(M):
                X_in[m].append(X[m].cpu().numpy())

            for m1 in range(M):
                for m2 in range(M):
                    if m1 != m2:
                        U_chunks[m1][m2].append(outputs['U'][:, m1, m2, :].cpu().numpy())
                        S_chunks[m1][m2].append(outputs['S_view'][:, m1, m2, :].cpu().numpy())
                        
            if batch_idx == 0:
                Labels_U = {k: v.detach().clone() for k, v in labels_u.items()}
                Labels_S = {k: v.detach().clone() for k, v in labels_s.items()}
                T_u = {k: v.detach().clone() for k, v in t_u.items()}
                T_s = {k: v.detach().clone() for k, v in t_s.items()}
            else:
                Labels_U = {k: torch.cat([Labels_U[k], labels_u[k]], dim=0) for k, v in labels_u.items()}
                Labels_S = {k: torch.cat([Labels_S[k], labels_s[k]], dim=0) for k, v in labels_s.items()}
                T_u = {k: torch.cat([T_u[k], t_u[k]], dim=0) for k, v in t_u.items()}
                T_s = {k: torch.cat([T_s[k], t_s[k]], dim=0) for k, v in t_s.items()}

    # U_final[m1][m2] will be (N, D) where N = total samples
    U_final = [[None for _ in range(M)] for _ in range(M)]
    S_final = [[None for _ in range(M)] for _ in range(M)]

    for m1 in range(M):
        for m2 in range(M):
            if m1 == m2:
                continue
            U_final[m1][m2] = np.concatenate(U_chunks[m1][m2], axis=0)
            S_final[m1][m2] = np.concatenate(S_chunks[m1][m2], axis=0)
    data_dict = {
        "X_in": [np.concatenate(X_in[m], axis=0) for m in range(M)],
        "t_u": T_u,
        "t_s": T_s,
        'U': U_final,
        'S': S_final,
        'Labels_U': {k: v.cpu().numpy() for k, v in Labels_U.items()},
        'Labels_S': {k: v.cpu().numpy() for k, v in Labels_S.items()}
    }
    return data_dict


@dataclass
class ProbeEvaluator:
    """
    Stateful wrapper for running linear and regression probes on learned representations.
    """
    linear_probe: Optional[Callable] = None
    regression_probe: Optional[Callable] = None
    
    # Stored state
    train_data_dict: Optional[Dict[str, Any]] = None
    val_data_dict: Optional[Dict[str, Any]] = None
    M: Optional[int] = None

    comp_keys: List[str] = field(default_factory=list)
    label_keys: List[str] = field(default_factory=list)

    _y_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    linear_probe_results: Optional[Dict[str, Any]] = None
    reg_probe_results: Optional[Dict[str, Any]] = None

    _pair_re = re.compile(r"^[us]_(\d+)(\d+)$|^[us]_(\d+)_(\d+)$")

    def _require_probe(self, name: str, fn: Optional[Callable]) -> Callable:
        if fn is None:
            raise RuntimeError(
                f"{name} is not set.\n"
                f"Pass `{name}` when constructing the class, e.g.:\n\n"
                f"  ProbeEvaluator({name}={name})"
            )
        return fn

    # Set the data for the evaluator
    def set_data(self, train_data_dict: Dict[str, Any], val_data_dict: Dict[str, Any], M: Optional[int] = None) -> "ProbeEvaluator":
        self.train_data_dict = train_data_dict
        self.val_data_dict = val_data_dict
        if M is not None:
            self.M = M

        self.comp_keys = list(train_data_dict["Labels_U"].keys()) + list(train_data_dict["Labels_S"].keys())
        self.label_keys = list(train_data_dict["Labels_U"].keys()) + list(train_data_dict["Labels_S"].keys())

        self._y_cache = {lab: (self.get_labels(train_data_dict, lab), self.get_labels(val_data_dict, lab))
                         for lab in self.label_keys}
        return self

    # Some helpers for the probes
    @classmethod
    def parse_pair(cls, key: str) -> Tuple[int, int]:
        m = cls._pair_re.match(key)
        if not m:
            # fallback to your original strict "u_12" indexing
            i = int(key[2]) - 1
            j = int(key[3]) - 1
            return i, j
        a = m.group(1) or m.group(3)
        b = m.group(2) or m.group(4)
        return int(a) - 1, int(b) - 1

    def get_features(self, data_dict: Dict[str, Any], comp_key: str) -> np.ndarray:
        return data_dict["t_s" if "s" in comp_key else "t_u"][comp_key]

    def get_latents(self, data_dict: Dict[str, Any], comp_key: str) -> np.ndarray:
        i, j = self.parse_pair(comp_key)
        if comp_key.startswith("u"):
            return data_dict["U"][i][j]
        elif comp_key.startswith("s"):
            return np.concatenate([data_dict["S"][i][j], data_dict["S"][j][i]], axis=-1)
        raise ValueError(f"Unknown component key: {comp_key}")

    @staticmethod
    def get_labels(data_dict: Dict[str, Any], label_key: str) -> np.ndarray:
        if label_key in data_dict["Labels_U"]:
            return data_dict["Labels_U"][label_key]
        if label_key in data_dict["Labels_S"]:
            return data_dict["Labels_S"][label_key]
        raise KeyError(f"Label key {label_key} not found in Labels_U or Labels_S")

    
    def calculate_linear_probe(self, train_data_dict=None, val_data_dict=None) -> Dict[str, Any]:
        train_data_dict = train_data_dict or self.train_data_dict
        val_data_dict = val_data_dict or self.val_data_dict
        
        if train_data_dict is None or val_data_dict is None:
            raise ValueError("train_data_dict/val_data_dict not set. Call set_data(...) or pass them in.")

        linear_probe_fn = self._require_probe("linear_probe", self.linear_probe)

        comp_keys = list(train_data_dict["Labels_U"].keys()) + list(train_data_dict["Labels_S"].keys())
        label_keys = list(train_data_dict["Labels_U"].keys()) + list(train_data_dict["Labels_S"].keys())

        acc = {lab: np.zeros(len(comp_keys), dtype=float) for lab in label_keys}
        mcc = {lab: np.zeros(len(comp_keys), dtype=float) for lab in label_keys}
        f1 = {lab: np.zeros(len(comp_keys), dtype=float) for lab in label_keys}
        recall = {lab: np.zeros(len(comp_keys), dtype=float) for lab in label_keys}

        y_cache = self._y_cache if (self._y_cache and label_keys == self.label_keys) else {
            lab: (self.get_labels(train_data_dict, lab), self.get_labels(val_data_dict, lab))
            for lab in label_keys
        }

        for c_idx, comp in enumerate(comp_keys):
            Xtr = self.get_latents(train_data_dict, comp)
            Xva = self.get_latents(val_data_dict, comp)

            for lab in label_keys:
                ytr, yva = y_cache[lab]
                results = linear_probe_fn(Xtr, ytr, Xva, yva)

                acc[lab][c_idx] = results["acc"]
                mcc[lab][c_idx] = results["mcc"]
                f1[lab][c_idx] = results["f1"]
                recall[lab][c_idx] = results["recall"]

        out = {"acc": acc, "mcc": mcc, "f1": f1, "recall": recall}
        self.linear_probe_results = out
        return out

    def calculate_reg_probe(self, train_data_dict=None, val_data_dict=None, M=None) -> Dict[str, Any]:
        train_data_dict = train_data_dict or self.train_data_dict
        val_data_dict = val_data_dict or self.val_data_dict
        M = M if M is not None else self.M
        if train_data_dict is None or val_data_dict is None or M is None:
            raise ValueError("train_data_dict/val_data_dict/M not set. Call set_data(..., M=...) or pass them in.")

        regression_probe_fn = self._require_probe("regression_probe", self.regression_probe)

        mse = {f"X_{i}": np.zeros((M,), dtype=float) for i in range(1, M + 1)}
        mae = {f"X_{i}": np.zeros((M,), dtype=float) for i in range(1, M + 1)}
        r2  = {f"X_{i}": np.zeros((M,), dtype=float) for i in range(1, M + 1)}

        for m1 in range(1, M + 1):
            for m2 in range(1, M + 1):
                if m1 == m2:
                    continue

                u_tr = self.get_latents(train_data_dict, f"u_{m1}{m2}")
                s_tr = train_data_dict["S"][m1 - 1][m2 - 1]
                Z_tr = np.concatenate([u_tr, s_tr], axis=-1)

                u_va = self.get_latents(val_data_dict, f"u_{m1}{m2}")
                s_va = val_data_dict["S"][m1 - 1][m2 - 1]
                Z_va = np.concatenate([u_va, s_va], axis=-1)

                X_u_tr = self.get_features(train_data_dict, f"u_{m1}{m2}")
                X_s_tr = self.get_features(train_data_dict, f"s_{min(m1, m2)}{max(m1, m2)}")
                X_tr = np.concatenate([X_u_tr, X_s_tr], axis=-1)

                X_u_va = self.get_features(val_data_dict, f"u_{m1}{m2}")
                X_s_va = self.get_features(val_data_dict, f"s_{min(m1, m2)}{max(m1, m2)}")
                X_va = np.concatenate([X_u_va, X_s_va], axis=-1)

                results = regression_probe_fn(Z_tr, X_tr, Z_va, X_va)

                s_tr_compl = train_data_dict["S"][m2 - 1][m1 - 1]
                Z_tr_compl = np.concatenate([u_tr, s_tr_compl], axis=-1)

                s_va_compl = val_data_dict["S"][m2 - 1][m1 - 1]
                Z_va_compl = np.concatenate([u_va, s_va_compl], axis=-1)

                results_compl = regression_probe_fn(Z_tr_compl, X_tr, Z_va_compl, X_va)

                mse[f"X_{m1}"][m2 - 1] = (results["mse"] + results_compl["mse"]) / 2
                mae[f"X_{m1}"][m2 - 1] = (results["mae"] + results_compl["mae"]) / 2
                r2[f"X_{m1}"][m2 - 1]  = (results["r2"]  + results_compl["r2"])  / 2

        out = {"mse": mse, "mae": mae, "r2": r2}
        self.reg_probe_results = out
        return out

    # Average of all the metrics
    def mean_metrics(self, linear_probe_results=None, reg_probe_results=None, M=None) -> Dict[str, Any]:
        linear_probe_results = linear_probe_results if linear_probe_results is not None else self.linear_probe_results
        reg_probe_results = reg_probe_results if reg_probe_results is not None else self.reg_probe_results
        M = M if M is not None else self.M

        if linear_probe_results is None or reg_probe_results is None or M is None:
            raise ValueError("Missing results/M. Run probes first or pass in results/M explicitly.")

        metrics_summary = {}

        components = linear_probe_results["acc"].keys()
        rev_components = {comp: idx for idx, comp in enumerate(components)}

        for metric_name, results_dict in linear_probe_results.items():
            all_u2u, all_u2s, all_s2s, all_s2u = [], [], [], []
            for comp_key, values in results_dict.items():
                i, j = self.parse_pair(comp_key)
                if comp_key.startswith("u"):
                    all_u2u.append(values[rev_components[f"u_{i+1}{j+1}"]])
                    all_u2s.append(values[rev_components[f"s_{min(i+1, j+1)}{max(i+1, j+1)}"]])
                elif comp_key.startswith("s"):
                    all_s2s.append(values[rev_components[f"s_{min(i+1, j+1)}{max(i+1, j+1)}"]])
                    all_s2u.append((values[rev_components[f"u_{i+1}{j+1}"]] + values[rev_components[f"u_{j+1}{i+1}"]]) / 2)

            metrics_summary[f"linear_probe/u2u_{metric_name}_mean"] = np.mean(np.array(all_u2u), dtype=np.float32)
            metrics_summary[f"linear_probe/u2u_{metric_name}_std"]  = np.std(np.array(all_u2u), dtype=np.float32)

            metrics_summary[f"linear_probe/u2s_{metric_name}_mean"] = np.mean(np.array(all_u2s), dtype=np.float32)
            metrics_summary[f"linear_probe/u2s_{metric_name}_std"]  = np.std(np.array(all_u2s), dtype=np.float32)

            metrics_summary[f"linear_probe/s2u_{metric_name}_mean"] = np.mean(np.array(all_s2u), dtype=np.float32)
            metrics_summary[f"linear_probe/s2u_{metric_name}_std"]  = np.std(np.array(all_s2u), dtype=np.float32)

            metrics_summary[f"linear_probe/s2s_{metric_name}_mean"] = np.mean(np.array(all_s2s), dtype=np.float32)
            metrics_summary[f"linear_probe/s2s_{metric_name}_std"]  = np.std(np.array(all_s2s), dtype=np.float32)

        reg_metrics = {key: {} for key in reg_probe_results.keys()}
        for metric_name, results_dict in reg_probe_results.items():
            for modality_key, values in results_dict.items():
                valid = values[values > 0]
                reg_metrics[metric_name][f"{modality_key}_mean"] = np.mean(valid, dtype=np.float32)
                reg_metrics[metric_name][f"{modality_key}_std"]  = np.std(valid, dtype=np.float32)

                metrics_summary[f"regression_probe/{modality_key}_{metric_name}_mean"] = reg_metrics[metric_name][f"{modality_key}_mean"]
                metrics_summary[f"regression_probe/{modality_key}_{metric_name}_std"]  = reg_metrics[metric_name][f"{modality_key}_std"]

            overall_mean = np.mean([val for key, val in reg_metrics[metric_name].items() if "mean" in key], dtype=np.float32)
            overall_std  = np.std([val for key, val in reg_metrics[metric_name].items() if "std" in key], dtype=np.float32)

            reg_metrics[metric_name]["overall_mean"] = overall_mean
            reg_metrics[metric_name]["overall_std"]  = overall_std

            metrics_summary[f"regression_probe/{metric_name}_mean"] = overall_mean
            metrics_summary[f"regression_probe/{metric_name}_std"]  = overall_std

        return metrics_summary

    def run_all(self) -> Dict[str, Any]:
        self.calculate_linear_probe()
        self.calculate_reg_probe()
        return self.mean_metrics()