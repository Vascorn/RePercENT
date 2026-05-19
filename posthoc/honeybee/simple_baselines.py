import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from training.main_honeybee import (
    DEFAULT_FILTER_CANCER_TYPES,
    _filter_dataset_by_cancer_types,
    _format_filter_cancer_types,
    _parse_filter_cancer_types,
)


HONEYBEE_MODALITIES = ["clinical_qwen", "pathology_qwen", "wsi", "molecular"]


def _build_summary_table_payload(summary_metrics):
    cancer_types = sorted(
        {
            str(cancer_type)
            for component_metrics in summary_metrics.values()
            for cancer_type in component_metrics["per_cancer_type"].keys()
        }
    )
    payload = {"columns": ["component", "overall", *cancer_types], "data": []}

    for component_name in sorted(summary_metrics.keys()):
        component_metrics = summary_metrics[component_name]
        row = [
            component_name,
            f'{component_metrics["overall"]:.4f}',
        ]

        for cancer_type in cancer_types:
            score = component_metrics["per_cancer_type"].get(cancer_type)
            if score is None:
                row.append("N/A")
            else:
                row.append(f'{score:.4f}')

        payload["data"].append(row)

    return payload


def _build_wandb_summary_table(summary_metrics):
    payload = _build_summary_table_payload(summary_metrics)
    table = wandb.Table(columns=[str(column) for column in payload["columns"]])

    for row in payload["data"]:
        table.add_data(*row)

    return table


def save_summary_report(summary_metrics, script_dir, output_name):
    summary_rows = []

    for component_name in sorted(summary_metrics.keys()):
        component_metrics = summary_metrics[component_name]
        summary_rows.append(
            {
                "component": component_name,
                "eval": "overall",
                "mean": component_metrics["overall"],
                "std": 0.0,
            }
        )

        for cancer_type in sorted(component_metrics["per_cancer_type"].keys()):
            score = component_metrics["per_cancer_type"].get(cancer_type)
            if score is None:
                continue
            summary_rows.append(
                {
                    "component": component_name,
                    "eval": cancer_type,
                    "mean": score,
                    "std": 0.0,
                }
            )

    summary_report_df = pd.DataFrame(summary_rows, columns=["component", "eval", "mean", "std"])
    summary_dir = os.path.join(script_dir, "summary_reports", "cancer_type_component_summary")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, output_name)
    summary_report_df.to_csv(summary_path, index=False)
    print(f"Saved cancer type component summary table to {summary_path}")


def _masked_mean(embeddings, mask):
    embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
    mask = torch.as_tensor(mask, dtype=torch.bool)

    while mask.ndim < embeddings.ndim:
        mask = mask.unsqueeze(-1)

    masked_embeddings = embeddings * mask.to(dtype=embeddings.dtype)
    valid_count = mask.to(dtype=embeddings.dtype).sum(dim=0).clamp_min(1.0)
    return masked_embeddings.sum(dim=0) / valid_count


def _fuse_modality_embeddings(embeddings, pad_mask):
    embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
    pad_mask = torch.as_tensor(pad_mask, dtype=torch.bool)

    if embeddings.ndim in {2, 3}:
        return _masked_mean(embeddings, pad_mask)

    if embeddings.ndim == 4:
        slide_embeddings = []
        slide_mask = pad_mask.any(dim=-1)
        for slide_idx in range(embeddings.shape[0]):
            if not bool(slide_mask[slide_idx]):
                continue
            slide_embeddings.append(_masked_mean(embeddings[slide_idx], pad_mask[slide_idx]))

        if not slide_embeddings:
            return torch.zeros(embeddings.shape[-1], dtype=embeddings.dtype)
        return torch.stack(slide_embeddings, dim=0).mean(dim=0)

    raise ValueError(f"Unsupported embedding shape for fusion: {tuple(embeddings.shape)}")


def _extract_dataset_features(dataset, modality_order=None):
    modality_order = modality_order or HONEYBEE_MODALITIES
    feature_store = {modality: [] for modality in modality_order}
    feature_store["all_modalities"] = []
    labels = []

    for sample_idx in range(len(dataset)):
        sample = dataset[sample_idx]
        modality_features = []

        for modality in modality_order:
            embeddings, _, pad_mask, has_data = sample[modality]
            if not has_data:
                raise ValueError(f"Missing modality {modality} for sample {sample_idx}.")

            fused_feature = _fuse_modality_embeddings(embeddings, pad_mask)
            feature_store[modality].append(fused_feature.numpy())
            modality_features.append(fused_feature)

        feature_store["all_modalities"].append(torch.cat(modality_features, dim=0).numpy())
        labels.append(str(sample["cancer_type"]))

    feature_store = {
        name: np.stack(features, axis=0)
        for name, features in feature_store.items()
    }
    return feature_store, np.asarray(labels)


def _extract_acc_per_cancer_type(y_pred, y_true, label_encoder):
    acc_per_cancer_type = {}
    for idx, cancer_type in enumerate(label_encoder.classes_):
        class_mask = y_true == idx
        if not np.any(class_mask):
            acc_per_cancer_type[cancer_type] = float("nan")
            continue
        acc_per_cancer_type[cancer_type] = float(np.mean(y_pred[class_mask] == y_true[class_mask]))
    return acc_per_cancer_type


def _evaluate_linear_probes(train_features, train_labels, test_features, test_labels, modality_order=None):
    modality_order = modality_order or HONEYBEE_MODALITIES

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    y_test = label_encoder.transform(test_labels)

    metrics = {}
    for feature_name in [*modality_order, "all_modalities"]:
        clf = LogisticRegression(max_iter=2000)
        clf.fit(train_features[feature_name], y_train)
        y_pred = clf.predict(test_features[feature_name])
        metrics[feature_name] = {
            "overall": float(accuracy_score(y_test, y_pred)),
            "per_cancer_type": _extract_acc_per_cancer_type(y_pred, y_test, label_encoder),
        }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Linear probes on raw Honeybee modality embeddings")
    parser.add_argument('--datasets_path', type=str, default="../../data/honeybee/datasets/", help='Path to the directory containing the Honeybee dataset tensors wrt to this script')
    parser.add_argument('--wsi_embedding_mode', type=str, choices=['slide', 'patch'], default='slide', help='Method for aggregating WSI embeddings, either slide level or patch level')
    parser.add_argument('--split_seed', type=int, default=42, help='Seed for reproducible dataset splits')
    parser.add_argument('--filter_cancer_types', nargs='+', default=DEFAULT_FILTER_CANCER_TYPES, help='Optional cancer types to keep, e.g. --filter_cancer_types TCGA-BRCA TCGA-LUAD or TCGA-BRCA,TCGA-LUAD. Should match training.')
    parser.add_argument('--log_to_wandb', type=bool, default=False, help='Whether to log results to wandb')
    args = parser.parse_args()
    filter_cancer_types = _parse_filter_cancer_types(args.filter_cancer_types)
    filter_cancer_types_label = _format_filter_cancer_types(filter_cancer_types)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    analysis_config_path = os.path.join(script_dir, "../..", "configs", "posthoc_analysis", "honeybee.yaml")
    with open(analysis_config_path, 'r') as f:
        analysis_config = yaml.safe_load(f)

    dataset_split = torch.load(
        os.path.join(script_dir, args.datasets_path, f"dataset_01_{args.wsi_embedding_mode}_split_{args.split_seed}.pt"),
        weights_only=False,
    )
    train_dataset = dataset_split['train']
    test_dataset = dataset_split['test']
    train_dataset = _filter_dataset_by_cancer_types(train_dataset, filter_cancer_types)
    test_dataset = _filter_dataset_by_cancer_types(test_dataset, filter_cancer_types)
    if filter_cancer_types is not None:
        if len(train_dataset) == 0:
            raise ValueError(f"No training samples found for cancer types: {filter_cancer_types}.")
        if len(test_dataset) == 0:
            raise ValueError(f"No test samples found for cancer types: {filter_cancer_types}.")
        print(
            f"Filtered cancer types {filter_cancer_types}: "
            f"{len(train_dataset)} train samples, {len(test_dataset)} test samples"
        )

    train_features, train_labels = _extract_dataset_features(train_dataset, modality_order=HONEYBEE_MODALITIES)
    test_features, test_labels = _extract_dataset_features(test_dataset, modality_order=HONEYBEE_MODALITIES)


    print("Evaluating linear probes on fixed pre-extracted embeddings...")
    cancer_subtype_metrics = _evaluate_linear_probes(
        train_features,
        train_labels,
        test_features,
        test_labels,
        modality_order=HONEYBEE_MODALITIES,
    )
    print(f"Probe metrics: {cancer_subtype_metrics}")
    save_summary_report(
        cancer_subtype_metrics,
        script_dir,
        output_name=f"simple_baselines_cancer_type_component_summary.csv",
    )

    if args.log_to_wandb:
        wandb.init(
            project=analysis_config["wandb"]["project"],
            name=f"simple_baselines_cancer_type_probe_{args.wsi_embedding_mode}",
            config={
                "split_seed": args.split_seed,
                "wsi_embedding_mode": args.wsi_embedding_mode,
                "modalities": HONEYBEE_MODALITIES,
                "fusion": "mean_within_modality_then_concat_across_modalities",
                "filter_cancer_types": filter_cancer_types_label,
            },
        )

        summary_table = _build_wandb_summary_table(cancer_subtype_metrics)

        wandb.log({
            "cancer_type_component_summary": summary_table
        })
        wandb.finish()


if __name__ == "__main__":
    main()
