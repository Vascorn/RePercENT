import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Subset
import wandb
import yaml

from posthoc.honeybee.helper_metrics import HONEYBEE_MODALITIES
from posthoc.plotting_config import apply_paper_plot_style, get_line_style, percent_formatter
from src.models.repercent import RePercENT
from src.utils.helpers import set_seed
from training.main_honeybee import (
    DEFAULT_FILTER_CANCER_TYPES,
    _filter_dataset_by_cancer_types,
    _format_filter_cancer_types,
    _parse_filter_cancer_types,
)
from training.train_repercent import make_model

apply_paper_plot_style()
WSI_MODALITY = "wsi"
MOL_MODALITY = "molecular"
METHOD_ORDER = [
    "RePercENT",
    "Late fusion + averaging",
    "Early fusion + mean imputation",
    "Early fusion + mask",
    "Early fusion + modality dropout",
]


def _has_modality(sample, modality):
    has_data = sample[modality][3]
    if isinstance(has_data, torch.Tensor):
        return bool(has_data.all().item())
    return bool(has_data)


def _filter_complete_wsi_mol(dataset):
    indices = [
        idx for idx in range(len(dataset))
        if _has_modality(dataset[idx], WSI_MODALITY) and _has_modality(dataset[idx], MOL_MODALITY)
    ]
    return Subset(dataset, indices)


def _masked_average(embeddings, pad_mask):
    embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
    pad_mask = torch.as_tensor(pad_mask, dtype=torch.bool)

    flat_embeddings = embeddings.reshape(-1, embeddings.shape[-1])
    flat_mask = pad_mask.reshape(-1)
    if not bool(flat_mask.any()):
        return torch.zeros(embeddings.shape[-1], dtype=torch.float32)
    return flat_embeddings[flat_mask].mean(dim=0)


def _extract_raw_wsi_mol_features(dataset):
    features = {"z_mol": [], "z_wsi": []}
    labels = []
    patient_ids = []

    for sample_idx in range(len(dataset)):
        sample = dataset[sample_idx]
        for modality in [MOL_MODALITY, WSI_MODALITY]:
            if not _has_modality(sample, modality):
                raise ValueError(f"Sample {sample_idx} is missing required modality {modality}.")

        mol_embeddings, _, mol_mask, _ = sample[MOL_MODALITY]
        wsi_embeddings, _, wsi_mask, _ = sample[WSI_MODALITY]
        features["z_mol"].append(_masked_average(mol_embeddings, mol_mask).numpy())
        features["z_wsi"].append(_masked_average(wsi_embeddings, wsi_mask).numpy())
        labels.append(str(sample["cancer_type"]))
        patient_ids.append(str(sample.get("patient_id", sample_idx)))

    return {
        key: np.stack(value, axis=0)
        for key, value in features.items()
    }, np.asarray(labels), np.asarray(patient_ids)


def _prepare_modality_batch(batch, modality, device):
    embeddings, _, pad_mask, has_data = batch[modality]
    if isinstance(has_data, torch.Tensor):
        if not bool(has_data.all().item()):
            raise ValueError(f"Batch contains missing {modality} data before simulated missingness.")
    elif not bool(has_data):
        raise ValueError(f"Batch contains missing {modality} data before simulated missingness.")

    mask = pad_mask.bool().to(device) if pad_mask is not None else None
    return embeddings.to(device), mask


def _source_pos_enc(model, source_modality_idx):
    if not getattr(model, "add_pos_encoding", False):
        return None

    m = source_modality_idx + 1
    p_idx = getattr(model, f"pair_idx_m{m}")
    t_idx = getattr(model, f"type_idx_m{m}")
    return model.pair_pos_enc[p_idx] + model.type_pos_enc[t_idx]


def _encode_source_components(model, batch, source_modality, target_modality, modality_order, device):
    source_idx = modality_order.index(source_modality)
    target_idx = modality_order.index(target_modality)
    x, mask = _prepare_modality_batch(batch, source_modality, device)
    encoded = model.disenEncoders[source_idx](x, mask=mask, pos_enc=_source_pos_enc(model, source_idx))

    source_number = source_idx + 1
    target_number = target_idx + 1
    unique = model.get_slot(encoded, source_number, f"U_{source_number}{target_number}")
    shared = model.get_slot(encoded, source_number, f"S_{source_number}{target_number}")
    return torch.cat([unique, shared], dim=-1)


def _collect_repercent_wsi_mol_features(loader, model, device, modality_order):
    features = {"D_mol": [], "D_wsi": []}
    labels = []

    model.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            print(f"Extracting RePercENT WSI/molecular features batch {batch_idx + 1}/{len(loader)}")
            features["D_mol"].append(
                _encode_source_components(model, batch, MOL_MODALITY, WSI_MODALITY, modality_order, device).cpu()
            )
            features["D_wsi"].append(
                _encode_source_components(model, batch, WSI_MODALITY, MOL_MODALITY, modality_order, device).cpu()
            )
            labels.extend([str(label) for label in batch["cancer_type"]])

    return {
        key: torch.cat(value, dim=0).numpy()
        for key, value in features.items()
    }, np.asarray(labels)


def _fit_probe(x_train, y_train, seed, max_iter):
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=max_iter,
        class_weight="balanced",
        random_state=seed,
    )
    clf.fit(x_train, y_train)
    return clf


def _decision_scores(clf, x):
    scores = clf.decision_function(x)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    return scores


def _average_available_scores(mol_scores, wsi_scores, mol_available, wsi_available):
    mol_available = mol_available.astype(np.float32)[:, None]
    wsi_available = wsi_available.astype(np.float32)[:, None]
    denom = np.clip(mol_available + wsi_available, 1.0, None)
    return (mol_scores * mol_available + wsi_scores * wsi_available) / denom


def _zero_masked_features(z_mol, z_wsi, mol_available, wsi_available):
    mol_available_float = mol_available.astype(np.float32)[:, None]
    wsi_available_float = wsi_available.astype(np.float32)[:, None]
    return np.concatenate(
        [
            z_mol * mol_available_float,
            z_wsi * wsi_available_float,
            mol_available_float,
            wsi_available_float,
        ],
        axis=1,
    )


def _imputed_concat_features(z_mol, z_wsi, mol_available, wsi_available, mol_fill, wsi_fill):
    z_mol_imp = np.where(mol_available[:, None], z_mol, mol_fill)
    z_wsi_imp = np.where(wsi_available[:, None], z_wsi, wsi_fill)
    return np.concatenate([z_mol_imp, z_wsi_imp], axis=1)


def _dropout_training_features(z_mol, z_wsi, y_train, seed, n_augments, drop_prob):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    n_samples = len(y_train)

    complete_mask = np.ones(n_samples, dtype=bool)
    xs.append(_zero_masked_features(z_mol, z_wsi, complete_mask, complete_mask))
    ys.append(y_train)

    for _ in range(n_augments):
        mol_available = rng.random(n_samples) > drop_prob
        wsi_available = rng.random(n_samples) > drop_prob
        both_missing = ~(mol_available | wsi_available)
        if np.any(both_missing):
            keep_mol = rng.random(np.sum(both_missing)) < 0.5
            mol_available[both_missing] = keep_mol
            wsi_available[both_missing] = ~keep_mol

        xs.append(_zero_masked_features(z_mol, z_wsi, mol_available, wsi_available))
        ys.append(y_train)

    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _fit_baseline_probes(raw_train, y_train, seed, max_iter, dropout_augments, dropout_prob):
    z_mol = raw_train["z_mol"]
    z_wsi = raw_train["z_wsi"]

    print(f"Fitting Raw Molecular probe with seed {seed} and max_iter {max_iter}")
    raw_mol_clf = _fit_probe(z_mol, y_train, seed, max_iter)
    print(f"Fitting Raw WSI probe with seed {seed} and max_iter {max_iter}")
    raw_wsi_clf = _fit_probe(z_wsi, y_train, seed, max_iter)
    print(f"Fitting Concat mean impute probe with seed {seed} and max_iter {max_iter}")
    concat_mean_clf = _fit_probe(np.concatenate([z_mol, z_wsi], axis=1), y_train, seed, max_iter)
    print(f"Fitting Masked fusion probe with seed {seed} and max_iter {max_iter}")
    complete_mask = np.ones(len(y_train), dtype=bool)
    masked_clf = _fit_probe(
        _zero_masked_features(z_mol, z_wsi, complete_mask, complete_mask),
        y_train,
        seed,
        max_iter,
    )
    probes = {
        "raw_mol": raw_mol_clf,
        "raw_wsi": raw_wsi_clf,
        "concat_mean": {
            "clf": concat_mean_clf,
            "mol_fill": z_mol.mean(axis=0, keepdims=True),
            "wsi_fill": z_wsi.mean(axis=0, keepdims=True),
        },
        "masked": masked_clf,
    }
    print(f"Fitting Modality dropout fusion probe with seed {seed} and max_iter {max_iter}")
    dropout_x, dropout_y = _dropout_training_features(
        z_mol,
        z_wsi,
        y_train,
        seed=seed,
        n_augments=dropout_augments,
        drop_prob=dropout_prob,
    )
    probes["dropout"] = _fit_probe(dropout_x, dropout_y, seed, max_iter)
    return probes


def _fit_repercent_probes(repercent_train, y_train, seed, max_iter):
    D_mol = repercent_train["D_mol"]
    D_wsi = repercent_train["D_wsi"]

    print(f"Fitting RePercENT probes with seed {seed} and max_iter {max_iter}")
    return {
        "mol": _fit_probe(D_mol, y_train, seed, max_iter),
        "wsi": _fit_probe(D_wsi, y_train, seed, max_iter),
    }


def _predict_all_methods(repercent_probes, repercent_test, baseline_probes, raw_test, mol_available, wsi_available):
    predictions = {}
    D_mol = repercent_test["D_mol"]
    D_wsi = repercent_test["D_wsi"]
    z_mol = raw_test["z_mol"]
    z_wsi = raw_test["z_wsi"]

    rep_mol_scores = _decision_scores(repercent_probes["mol"], D_mol)
    rep_wsi_scores = _decision_scores(repercent_probes["wsi"], D_wsi)
    predictions["RePercENT"] = _average_available_scores(
        rep_mol_scores, rep_wsi_scores, mol_available, wsi_available
    ).argmax(axis=1)

    raw_mol_scores = _decision_scores(baseline_probes["raw_mol"], z_mol)
    raw_wsi_scores = _decision_scores(baseline_probes["raw_wsi"], z_wsi)
    predictions["Late fusion + averaging"] = _average_available_scores(
        raw_mol_scores, raw_wsi_scores, mol_available, wsi_available
    ).argmax(axis=1)

    concat_probe = baseline_probes["concat_mean"]
    concat_x = _imputed_concat_features(
        z_mol,
        z_wsi,
        mol_available,
        wsi_available,
        concat_probe["mol_fill"],
        concat_probe["wsi_fill"],
    )
    predictions["Early fusion + mean imputation"] = _decision_scores(concat_probe["clf"], concat_x).argmax(axis=1)

    masked_x = _zero_masked_features(z_mol, z_wsi, mol_available, wsi_available)
    predictions["Early fusion + mask"] = _decision_scores(baseline_probes["masked"], masked_x).argmax(axis=1)
    predictions["Early fusion + modality dropout"] = _decision_scores(baseline_probes["dropout"], masked_x).argmax(axis=1)

    return predictions


def _metric_value(y_true, y_pred, metric_name):
    if metric_name == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    if metric_name == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro"))
    raise ValueError(f"Unsupported metric: {metric_name}")


def _simulate_missingness_curves(
    repercent_probes,
    repercent_test,
    baseline_probes,
    raw_test,
    y_test,
    seed_label,
    drop_rates,
    num_repeats,
    metric_name,
):
    rows = []
    n_test = len(y_test)

    for panel_name, dropped_modality in [
        ("Panel A: molecular available, drop WSI", WSI_MODALITY),
        ("Panel B: WSI available, drop molecular", MOL_MODALITY),
    ]:
        for drop_rate in drop_rates:
            for repeat in range(num_repeats):
                rng = np.random.default_rng(seed_label * 10000 + repeat * 1000 + int(round(drop_rate * 1000)))
                dropped_available = rng.random(n_test) > drop_rate

                if dropped_modality == WSI_MODALITY:
                    mol_available = np.ones(n_test, dtype=bool)
                    wsi_available = dropped_available
                else:
                    mol_available = dropped_available
                    wsi_available = np.ones(n_test, dtype=bool)

                predictions = _predict_all_methods(
                    repercent_probes,
                    repercent_test,
                    baseline_probes,
                    raw_test,
                    mol_available,
                    wsi_available,
                )
                for method_name in METHOD_ORDER:
                    rows.append({
                        "seed": seed_label,
                        "repeat": repeat,
                        "panel": panel_name,
                        "dropped_modality": dropped_modality,
                        "drop_rate": float(drop_rate),
                        "method": method_name,
                        "metric": metric_name,
                        "value": _metric_value(y_test, predictions[method_name], metric_name),
                    })

    return rows


def _summarize_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["panel"], row["method"], row["drop_rate"], row["metric"])].append(row["value"])

    summary = []
    for (panel, method, drop_rate, metric), values in sorted(grouped.items()):
        values = np.asarray(values, dtype=np.float64)
        summary.append({
            "panel": panel,
            "method": method,
            "drop_rate": float(drop_rate),
            "metric": metric,
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "n": int(len(values)),
        })
    return summary


def _write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_summary_csv(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "panel": row["panel"],
                "method": row["method"],
                "drop_rate": float(row["drop_rate"]),
                "metric": row["metric"],
                "mean": float(row["mean"]),
                "std": float(row["std"]),
                "n": int(row["n"]),
            })
    return rows


def _missingness_output_paths(script_dir, args):
    figure_dir = os.path.join(script_dir, "figures", "missingness")
    csv_dir = os.path.join(script_dir, "summary_reports", "missingess_summary")
    run_stem = f"wsi_mol_{args.wsi_embedding_mode}_{args.metric}_split{args.split_seed}"
    csv_stem = f"wsi_mol_{args.wsi_embedding_mode}_{args.metric}_split{args.split_seed}"
    return {
        "csv_dir": csv_dir,
        "figure_dir": figure_dir,
        "run_stem": run_stem,
        "raw_csv": os.path.join(csv_dir, f"{csv_stem}_raw.csv"),
        "summary_csv": os.path.join(csv_dir, f"{csv_stem}_summary.csv"),
        "figure_stem": os.path.join(figure_dir, run_stem),
    }


def _reference_value(summary_rows, panel, method, drop_rate):
    candidates = [
        row for row in summary_rows
        if row["panel"] == panel and row["method"] == method and np.isclose(row["drop_rate"], drop_rate)
    ]
    if not candidates:
        return None
    return candidates[0]["mean"]


def _repercent_single_modality_references(summary_rows):
    mol_only = _reference_value(
        summary_rows,
        "Panel A: molecular available, drop WSI",
        "RePercENT",
        1.0,
    )
    wsi_only = _reference_value(
        summary_rows,
        "Panel B: WSI available, drop molecular",
        "RePercENT",
        1.0,
    )
    return {
        "Molecular only": mol_only,
        "WSI only": wsi_only,
    }


def _plot_missingness(summary_rows, output_stem, metric_name):
    panel_specs = [
        (
            "panel_a",
            "Panel A: molecular available, drop WSI",
            "Robustness to missing WSI data",
        ),
        (
            "panel_b",
            "Panel B: WSI available, drop molecular",
            "Robustness to missing Molecular data",
        ),
    ]
    y_label = "Balanced accuracy" if metric_name == "balanced_accuracy" else "Macro-F1"
    references = _repercent_single_modality_references(summary_rows)
    reference_styles = {
        "Molecular only": "--",
        "WSI only": "dotted",
    }

    all_means = [row["mean"] for row in summary_rows]
    all_means.extend(value for value in references.values() if value is not None)
    y_min = max(0.0, min(all_means) - 0.04) if all_means else 0.0
    y_max = min(1.0, max(all_means) + 0.04) if all_means else 1.0

    output_paths = {}
    for panel_key, panel, title in panel_specs:
        fig, ax = plt.subplots(figsize=(6, 5))
        panel_rows = [row for row in summary_rows if row["panel"] == panel]
        for method_idx, method in enumerate(METHOD_ORDER):
            method_rows = sorted([row for row in panel_rows if row["method"] == method], key=lambda x: x["drop_rate"])
            if not method_rows:
                continue
            x = np.asarray([row["drop_rate"] for row in method_rows])
            y = np.asarray([row["mean"] for row in method_rows])
            y_std = np.asarray([row["std"] for row in method_rows])
            line_style = get_line_style(method, index=method_idx)
            ax.plot(x, y, label=method, **line_style)
            ax.fill_between(x, y - y_std, y + y_std, alpha=0.15, color=line_style["color"])

        for label, value in references.items():
            if value is None:
                continue
            ax.axhline(
                value,
                color="gray",
                linewidth=2,
                linestyle=reference_styles[label],
                zorder=0,
            )
            ax.annotate(
                label,
                xy=(0.92, value),
                xycoords=ax.get_yaxis_transform(),
                xytext=(0.82, value - 0.03),
                textcoords=ax.get_yaxis_transform(),
                color="gray",
                fontsize=9,
                fontweight="semibold",
                ha="right",
                va="top",
                arrowprops={
                    "arrowstyle": '-|>',
                    "color": "gray",
                    "linewidth": 0.3,
                    "connectionstyle": "arc3,rad=0.25",
                    "shrinkA": 2,
                    "shrinkB": 2,
                },
            )

        ax.set_title(title)
        ax.set_xlabel("WSI missing rate" if "WSI" in panel else "Molecular missing rate")
        ax.set_ylabel(y_label)
        ax.set_xticks(np.linspace(0.0, 1.0, 6))
        ax.xaxis.set_major_formatter(percent_formatter)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower left", frameon=True, framealpha=0.9, edgecolor="none")
        
        output_path = f"{output_stem}_{panel_key}.pdf"
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        output_paths[panel_key] = output_path

    return output_paths


def _build_summary_table(summary_rows):
    table = wandb.Table(columns=["panel", "method", "drop_rate", "metric", "mean", "std", "n"])
    for row in summary_rows:
        table.add_data(
            row["panel"],
            row["method"],
            row["drop_rate"],
            row["metric"],
            row["mean"],
            row["std"],
            row["n"],
        )
    return table


def _load_repercent_model(model_config, data_config, checkpoint_path, project_root, device):
    M = data_config["create_data"]["M"]
    disen_encoders = [
        make_model(model_config, data_config, modality=m + 1, M=M)
        for m in range(M)
    ]
    model = RePercENT(
        M=M,
        disenEncoder=disen_encoders,
        disen_mapping=model_config["repercent"]["disen_mapping"],
        vmfkappa=model_config["repercent"].get("vmfkappa", 1e3),
    ).to(device)

    state_dict = torch.load(os.path.join(project_root, checkpoint_path), map_location=device)
    model.load_state_dict(state_dict["model_state_dict"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Honeybee missing-modality cancer-type probes for WSI and molecular data")
    parser.add_argument("--datasets_path", type=str, default="../../data/honeybee/datasets/", help="Path to Honeybee dataset tensors relative to this script")
    parser.add_argument("--wsi_embedding_mode", type=str, choices=["slide", "patch"], default="slide", help="WSI embedding mode")
    parser.add_argument("--split_seed", type=int, default=42, help="Fixed train/test split seed")
    parser.add_argument("--base_seed", type=int, default=2, help="Base seed used for probes and missingness repeats")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for RePercENT feature extraction")
    parser.add_argument("--filter_cancer_types", nargs="+", default=DEFAULT_FILTER_CANCER_TYPES, help="Cancer types to keep")
    parser.add_argument("--metric", type=str, choices=["balanced_accuracy", "macro_f1"], default="balanced_accuracy", help="Missingness curve metric")
    parser.add_argument("--num_drop_rates", type=int, default=11, help="Number of evenly spaced drop rates between 0 and 1")
    parser.add_argument("--num_repeats", type=int, default=10, help="Number of missingness masks per drop rate")
    parser.add_argument("--max_iter", type=int, default=10000, help="Maximum iterations for LogisticRegression probes")
    parser.add_argument("--dropout_augments", type=int, default=4, help="Number of random modality-dropout copies added to complete training data")
    parser.add_argument("--modality_dropout_prob", type=float, default=0.5, help="Per-modality dropout probability for the modality-dropout fusion baseline")
    parser.add_argument("--max_seeds", type=int, default=None, help="Optional cap on RePercENT checkpoints to evaluate")
    parser.add_argument("--log_to_wandb", action=argparse.BooleanOptionalAction, default=True, help="Whether to log results and figures to Weights & Biases")
    parser.add_argument(
        "--plot_from_summary_csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If the expected summary CSV exists, only regenerate the missingness panel PDFs from it",
    )
    args = parser.parse_args()

    filter_cancer_types = _parse_filter_cancer_types(args.filter_cancer_types)
    filter_cancer_types_label = _format_filter_cancer_types(filter_cancer_types)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_paths = _missingness_output_paths(script_dir, args)
    if args.plot_from_summary_csv and os.path.exists(output_paths["summary_csv"]):
        summary_rows = _read_summary_csv(output_paths["summary_csv"])
        figure_paths = _plot_missingness(summary_rows, output_paths["figure_stem"], args.metric)
        print(f"Loaded summary results from {output_paths['summary_csv']}")
        print(f"Saved missingness panel A figure to {figure_paths['panel_a']}")
        print(f"Saved missingness panel B figure to {figure_paths['panel_b']}")
        return
    if args.plot_from_summary_csv:
        print(f"Summary CSV not found at {output_paths['summary_csv']}; running full evaluation.")

    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config_path = os.path.join(script_dir, "../..", "configs", "data", "honeybee_data.yaml")
    model_config_path = os.path.join(script_dir, "../..", "configs", "model", "repercent_honeybee.yaml")
    analysis_config_path = os.path.join(script_dir, "../..", "configs", "posthoc_analysis", "honeybee.yaml")
    with open(data_config_path, "r") as f:
        data_config = yaml.safe_load(f)
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)
    with open(analysis_config_path, "r") as f:
        analysis_config = yaml.safe_load(f)

    modality_order = data_config["create_data"].get("modalities", HONEYBEE_MODALITIES)
    if WSI_MODALITY not in modality_order or MOL_MODALITY not in modality_order:
        raise ValueError(f"Expected both {WSI_MODALITY} and {MOL_MODALITY} in modality order: {modality_order}")

    dataset_split = torch.load(
        os.path.join(script_dir, args.datasets_path, f"dataset_01_{args.wsi_embedding_mode}_split_{args.split_seed}.pt"),
        weights_only=False,
    )
    train_dataset = _filter_dataset_by_cancer_types(dataset_split["train"], filter_cancer_types)
    test_dataset = _filter_dataset_by_cancer_types(dataset_split["test"], filter_cancer_types)
    train_dataset = _filter_complete_wsi_mol(train_dataset)
    test_dataset = _filter_complete_wsi_mol(test_dataset)
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError("No complete WSI/molecular patients remain after filtering.")

    print(
        f"Missingness evaluation on complete WSI/molecular patients: "
        f"{len(train_dataset)} train, {len(test_dataset)} test"
    )

    raw_train, train_labels, _ = _extract_raw_wsi_mol_features(train_dataset)
    raw_test, test_labels, _ = _extract_raw_wsi_mol_features(test_dataset)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    y_test = label_encoder.transform(test_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    drop_rates = np.linspace(0.0, 1.0, args.num_drop_rates)

    checkpoint_paths = analysis_config["repercent"]["checkpoints"]
    if args.max_seeds is not None:
        checkpoint_paths = checkpoint_paths[:args.max_seeds]

    all_rows = []
    for seed_idx, checkpoint_path in enumerate(checkpoint_paths):
        temp_seed = args.base_seed + seed_idx
        print(f"Evaluating RePercENT checkpoint seed {temp_seed}: {checkpoint_path}")
        set_seed(temp_seed)
        model = _load_repercent_model(model_config, data_config, checkpoint_path, project_root, device)
        repercent_train, repercent_train_labels = _collect_repercent_wsi_mol_features(
            train_loader, model, device, modality_order
        )
        repercent_test, repercent_test_labels = _collect_repercent_wsi_mol_features(
            test_loader, model, device, modality_order
        )

        if not np.array_equal(train_labels, repercent_train_labels):
            raise ValueError("Train label order mismatch between raw and RePercENT features.")
        if not np.array_equal(test_labels, repercent_test_labels):
            raise ValueError("Test label order mismatch between raw and RePercENT features.")

        repercent_probes = _fit_repercent_probes(repercent_train, y_train, temp_seed, args.max_iter)
        baseline_probes = _fit_baseline_probes(
            raw_train,
            y_train,
            seed=temp_seed,
            max_iter=args.max_iter,
            dropout_augments=args.dropout_augments,
            dropout_prob=args.modality_dropout_prob,
        )

        all_rows.extend(
            _simulate_missingness_curves(
                repercent_probes,
                repercent_test,
                baseline_probes,
                raw_test,
                y_test,
                seed_label=temp_seed,
                drop_rates=drop_rates,
                num_repeats=args.num_repeats,
                metric_name=args.metric,
            )
        )

    summary_rows = _summarize_rows(all_rows)

    os.makedirs(output_paths["csv_dir"], exist_ok=True)
    os.makedirs(output_paths["figure_dir"], exist_ok=True)
    _write_csv(output_paths["raw_csv"], all_rows, ["seed", "repeat", "panel", "dropped_modality", "drop_rate", "method", "metric", "value"])
    _write_csv(output_paths["summary_csv"], summary_rows, ["panel", "method", "drop_rate", "metric", "mean", "std", "n"])
    figure_paths = _plot_missingness(summary_rows, output_paths["figure_stem"], args.metric)

    if args.log_to_wandb:
        wandb.init(
            project=analysis_config["wandb"]["project"],
            name=output_paths["run_stem"],
            config={
                "split_seed": args.split_seed,
                "base_seed": args.base_seed,
                "n_repercent_checkpoints": len(checkpoint_paths),
                "wsi_embedding_mode": args.wsi_embedding_mode,
                "filter_cancer_types": filter_cancer_types_label,
                "metric": args.metric,
                "num_drop_rates": args.num_drop_rates,
                "num_repeats": args.num_repeats,
                "dropout_augments": args.dropout_augments,
                "modality_dropout_prob": args.modality_dropout_prob,
                "feature_standardization": "none; raw foundation-model embeddings/components are used directly",
                "concat_missing_imputation": "raw train-set modality mean",
                "probe_class_weight": "balanced",
                "modalities_used_for_probes": [WSI_MODALITY, MOL_MODALITY],
                "all_model_modalities": modality_order,
            },
        )
        wandb.log({
            "missingness_summary": _build_summary_table(summary_rows)
        })
        wandb.save(output_paths["raw_csv"])
        wandb.save(output_paths["summary_csv"])
        wandb.finish()

        print(f"Saved missingness panel A figure to {figure_paths['panel_a']}")
        print(f"Saved missingness panel B figure to {figure_paths['panel_b']}")
        print(f"Saved raw results to {output_paths['raw_csv']}")
        print(f"Saved summary results to {output_paths['summary_csv']}")


if __name__ == "__main__":
    main()
