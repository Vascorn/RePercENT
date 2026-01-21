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
import re
from typing import Any, Callable, Dict, Optional, Tuple
from dataclasses import dataclass, field

def plot_confusion_matrix(linear_probe_acc, labels: List= ['labels_1', 'labels_2', 'labels_s'], components: List= ['u_12', 'u_21', 's']):

    fig, axes = plt.subplots(1, 1, figsize=(12, 10))
    

    # Extract the arrays from the dictionary in the specified order
    arrays_to_stack = [linear_probe_acc[key] for key in components]

    # Stack the arrays vertically to create the 4x3 matrix
    result_matrix = np.vstack(arrays_to_stack)
    cm = result_matrix
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=components, ax=axes)
    axes.set_title('Linear Probe Accuracy for Components')
    axes.set_xlabel('Labels')
    axes.set_ylabel('Components')

    plt.tight_layout()
    plt.show()
    return fig


def plot_pairwise_confusion_matrices(linear_probe_acc, M, components: List= ['u_12', 'u_21', 's'], pairs: List= None):
    """
    Plot M*(M-1)/2 confusion matrices, one per modality pair.

    Rows: labels
    Columns: components
    Values: linear probe accuracy
    """

    # reconstruct component & label keys (same set)
    label_keys = components.copy()
    comp_keys = components.copy()

    # build full accuracy matrix (rows=labels, cols=components)
    A = np.stack([linear_probe_acc[k] for k in label_keys], axis=0)
    x_shape = M if M % 2 else M // 2
    y_shape = M - 1 if (M - 1) % 2 else M // 2
    
    x_shape, y_shape = (y_shape, x_shape) if x_shape > y_shape else (x_shape, y_shape)
    fig, axes = plt.subplots(x_shape, y_shape, figsize=(5 * y_shape, 6 * x_shape))
    
    for pair_id, (i, j) in enumerate(pairs):
        
        pair_name = f"{i+1} vs {j+1}"
        
        # columns: u_ij, u_ji, s_ij
        col_keys = [f"u_{i+1}{j+1}", f"u_{j+1}{i+1}", f"s_{i+1}{j+1}"]
        col_idx = [comp_keys.index(k) for k in col_keys]

        # rows: same pairwise labels
        row_keys = col_keys
        row_idx = [label_keys.index(k) for k in row_keys]

        submat = A[np.ix_(row_idx, col_idx)]


        axes_id_x = pair_id // y_shape
        axes_id_y = pair_id % y_shape

        ax = axes[axes_id_x, axes_id_y] if x_shape > 1 and y_shape > 1 else axes[max(axes_id_x, axes_id_y)]
        sns.heatmap(
            submat,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=col_keys,
            yticklabels=row_keys,
            cbar=True,
            ax=ax
        )
        ax.set_title(f"Linear Probe – Pairwise Confusion ({pair_name})")
        ax.set_xlabel("Components")
        ax.set_ylabel("Labels")
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

def non_linear_probe(train_data, train_labels, test_data, test_labels):

    input_dim = train_data.shape[1]
 
    output_dim = len(np.unique(train_labels))

    model = nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.ReLU())

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Convert data to PyTorch tensors
    train_data_tensor = torch.FloatTensor(train_data)
    train_labels_tensor = torch.LongTensor(train_labels)
    test_data_tensor = torch.FloatTensor(test_data)
    test_labels_tensor = torch.LongTensor(test_labels)

    # Training loop
    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_data_tensor)
        loss = criterion(outputs, train_labels_tensor)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_data_tensor)
        _, predicted = torch.max(test_outputs.data, 1)
        total = test_labels_tensor.size(0)
        correct = (predicted == test_labels_tensor).sum().item()
        acc = (correct / total) * 100  # Convert to percentage
        mcc = matthews_corrcoef(test_labels_tensor.cpu(), predicted.cpu())
        f1 = f1_score(test_labels_tensor.cpu(), predicted.cpu(), average='weighted')
        recall = recall_score(test_labels_tensor.cpu(), predicted.cpu(), average='weighted')

    return {"acc": acc, "mcc": mcc, "f1": f1, "recall": recall, "predicted": predicted.cpu().numpy(), "true_labels": test_labels_tensor.cpu().numpy()}


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

def extract_latents_and_labels_2m(model, loader, device):
    model.eval()
    with torch.no_grad():
        for batch_idx, (data_m1, data_m2, labels_1, labels_2, labels_s) in enumerate(loader):
            dim_shape = data_m1.shape[-1] # dimension of original Z1 or Z2 
            data_m1 = data_m1.to(device)
            data_m2 = data_m2.to(device)
            labels_1 = labels_1.to(device)
            labels_2 = labels_2.to(device)
            labels_s = labels_s.to(device)
            
            outputs = model(data_m1, data_m2)
            u_12 = outputs['Z1'][0]
            s_21 = outputs['Z2'][1]
            u_21 = outputs['Z2'][0]
            s_12 = outputs['Z1'][1]
            
            if batch_idx == 0:
                # input data
                all_x12 = data_m1[:, 0, :dim_shape // 2]
                all_x21 = data_m2[:, 0, :dim_shape // 2]
                all_xs12 = data_m1[:, 0, dim_shape // 2:]
                all_xs21 = data_m2[:, 0, dim_shape // 2:]
                # latents
                all_u12 = u_12
                all_s21 = s_21
                all_u21 = u_21
                all_s12 = s_12
                all_labels_1 = labels_1
                all_labels_2 = labels_2
                all_labels_s = labels_s
            else:
                all_x12 = torch.cat([all_x12, data_m1[:, 0, :dim_shape // 2]], dim=0)
                all_x21 = torch.cat([all_x21, data_m2[:, 0, :dim_shape // 2]], dim=0)
                all_xs12 = torch.cat([all_xs12, data_m1[:, 0, dim_shape // 2:]], dim=0)
                all_xs21 = torch.cat([all_xs21, data_m2[:, 0, dim_shape // 2:]], dim=0)
                all_u12 = torch.cat([all_u12, u_12], dim=0)
                all_s21 = torch.cat([all_s21, s_21], dim=0)
                all_u21 = torch.cat([all_u21, u_21], dim=0)
                all_s12 = torch.cat([all_s12, s_12], dim=0)
                all_labels_1 = torch.cat([all_labels_1, labels_1], dim=0)
                all_labels_2 = torch.cat([all_labels_2, labels_2], dim=0)
                all_labels_s = torch.cat([all_labels_s, labels_s], dim=0)
    data_dict = {
        'x_12': all_x12.cpu().numpy(),
        'x_21': all_x21.cpu().numpy(),
        'xs_12': all_xs12.cpu().numpy(),
        'xs_21': all_xs21.cpu().numpy(),
        'u_12': all_u12.cpu().numpy(),
        's_21': all_s21.cpu().numpy(),
        'u_21': all_u21.cpu().numpy(),
        's_12': all_s12.cpu().numpy(),
        'labels_1': all_labels_1.cpu().numpy(),
        'labels_2': all_labels_2.cpu().numpy(),
        'labels_s': all_labels_s.cpu().numpy()}
    return data_dict

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
            
            outputs = model(X)

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