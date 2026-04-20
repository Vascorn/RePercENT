import argparse
import csv
import json
import os
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb

from posthoc.plotting_config import apply_paper_plot_style
from posthoc.honeybee.helper_metrics import HONEYBEE_MODALITIES, get_honeybee_modality_short_name
from training.main_honeybee import DEFAULT_FILTER_CANCER_TYPES, _parse_filter_cancer_types

apply_paper_plot_style()


def _sanitize_name(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def _extract_float(value):
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()
    if not text or text.lower() == "n/a":
        return np.nan

    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return float(match.group(0)) if match else np.nan


def _load_table_from_json(path):
    with open(path, 'r') as f:
        payload = json.load(f)
    return payload['columns'], payload['data']


def _build_run_path(entity, project, run_name):
    entity = str(entity).strip()
    project = str(project).strip()
    run_name = str(run_name).strip()
    if not entity or not project or not run_name:
        raise ValueError('Entity, project, and run name must all be provided.')
    return f'{entity}/{project}/{run_name}'


def _artifact_matches_table_key(artifact, table_key):
    name = str(getattr(artifact, 'name', ''))
    aliases = [str(alias) for alias in getattr(artifact, 'aliases', [])]
    return table_key in name or any(table_key in alias for alias in aliases)


def _download_wandb_table(entity, project, run_name, table_key):
    api = wandb.Api()
    run = api.run(_build_run_path(entity, project, run_name))

    matching_artifact = None
    for artifact in run.logged_artifacts():
        if _artifact_matches_table_key(artifact, table_key):
            matching_artifact = artifact
            break

    if matching_artifact is None:
        available = [str(getattr(artifact, 'name', '')) for artifact in run.logged_artifacts()]
        raise KeyError(
            f"Run {run.path} does not have a logged artifact for table key {table_key!r}. "
            f"Available artifacts: {available}"
        )

    table = matching_artifact.get(table_key)
    dataframe = table.get_dataframe()
    return list(dataframe.columns), dataframe.values.tolist()


def _load_table(table_json_path=None, entity=None, project=None, run_name=None, table_key='cancer_type_component_summary'):
    if table_json_path is not None:
        return _load_table_from_json(table_json_path)
    if run_name is not None:
        return _download_wandb_table(entity, project, run_name, table_key)
    raise ValueError('Either a local table JSON path or entity/project/run_name must be provided.')


def _table_to_metric_maps(columns, rows):
    if 'component' not in columns:
        raise ValueError(f"Expected a 'component' column, but got columns: {columns}")

    component_idx = columns.index('component')
    metric_columns = [column for column in columns if column not in {'component', 'overall'}]
    metric_indices = {column: columns.index(column) for column in metric_columns}

    metrics = {}
    raw_metrics = {}
    for row in rows:
        component_name = str(row[component_idx])
        metrics[component_name] = {
            cancer_type: _extract_float(row[idx])
            for cancer_type, idx in metric_indices.items()
        }
        raw_metrics[component_name] = {
            cancer_type: row[idx]
            for cancer_type, idx in metric_indices.items()
        }

    return metrics, raw_metrics, metric_columns


def _resolve_selected_cancer_types(disent_cancer_types, baseline_cancer_types, requested_cancer_types=None):
    shared_cancer_types = [cancer_type for cancer_type in disent_cancer_types if cancer_type in baseline_cancer_types]
    if not shared_cancer_types:
        raise ValueError('No shared cancer type columns were found between the two tables.')

    if not requested_cancer_types:
        return shared_cancer_types

    selected_cancer_types = list(dict.fromkeys(str(cancer_type) for cancer_type in requested_cancer_types))
    missing_cancer_types = sorted(set(selected_cancer_types) - set(shared_cancer_types))
    if missing_cancer_types:
        raise ValueError(
            f'Requested cancer types were not found in both tables: {missing_cancer_types}. '
            f'Shared cancer types: {shared_cancer_types}'
        )
    return selected_cancer_types


def _resolve_table_json_path(table_json_paths, table_keys, table_idx, label):
    if table_json_paths is None:
        return None

    if len(table_json_paths) == 1:
        if len(table_keys) > 1:
            raise ValueError(
                f'Only one {label} table JSON was provided for multiple table keys: {table_keys}. '
                f'Provide one JSON path per table key, in the same order.'
            )
        return table_json_paths[0]

    if len(table_json_paths) != len(table_keys):
        raise ValueError(
            f'Expected {len(table_keys)} {label} table JSON paths, one for each table key, '
            f'but got {len(table_json_paths)}: {table_json_paths}'
        )
    return table_json_paths[table_idx]


def _metric_label_for_table(table_key):
    if table_key == 'survival_analysis_component_summary':
        return 'c-index'
    if table_key == 'cancer_type_component_summary':
        return 'accuracy'
    return 'metric'


def _normalize_cli_values(values):
    if values is None:
        return None

    normalized = []
    for value in values:
        parts = str(value).replace(',', ' ').split()
        for part in parts:
            part = part.strip().strip('[](){}').strip("'\"")
            if part:
                normalized.append(part)
    return normalized


def _get_component_rows_for_modality(component_metrics, modality_name):
    short_name = get_honeybee_modality_short_name(modality_name)
    prefixes = (f'U_{short_name}_', f'S_{short_name}_', f'D_{short_name}_')
    return sorted(component_name for component_name in component_metrics.keys() if component_name.startswith(prefixes))


def _compute_delta_matrix(disent_metrics, baseline_metrics, modality_name, cancer_types):
    baseline_row = baseline_metrics.get(modality_name)
    if baseline_row is None:
        raise KeyError(
            f"Baseline table does not contain modality row {modality_name!r}. Available rows: {sorted(baseline_metrics.keys())}"
        )

    component_rows = _get_component_rows_for_modality(disent_metrics, modality_name)
    if not component_rows:
        return [], np.empty((0, len(cancer_types)), dtype=np.float32)

    delta = np.full((len(component_rows), len(cancer_types)), np.nan, dtype=np.float32)
    for row_idx, component_name in enumerate(component_rows):
        component_scores = disent_metrics[component_name]
        for col_idx, cancer_type in enumerate(cancer_types):
            delta[row_idx, col_idx] = component_scores.get(cancer_type, np.nan) - baseline_row.get(cancer_type, np.nan)

    return component_rows, delta


def _average_metric(metric_row, cancer_types):
    values = np.asarray([metric_row.get(cancer_type, np.nan) for cancer_type in cancer_types], dtype=np.float32)
    if np.all(np.isnan(values)):
        return np.nan
    return float(np.nanmean(values))


def _format_table_cell(value):
    if value is None:
        return 'N/A'
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return 'N/A'
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f'{float(value):.4f}'
    return str(value)


def _best_component_for_modality(disent_metrics, modality_name, cancer_types):
    component_rows = _get_component_rows_for_modality(disent_metrics, modality_name)
    if not component_rows:
        return None, None, np.nan

    best_component = None
    best_metrics = None
    best_average = -np.inf
    for component_name in component_rows:
        component_metrics = disent_metrics[component_name]
        average_metric = _average_metric(component_metrics, cancer_types)
        if np.isnan(average_metric):
            continue
        if average_metric > best_average:
            best_component = component_name
            best_metrics = component_metrics
            best_average = average_metric

    if best_component is None:
        return None, None, np.nan
    return best_component, best_metrics, best_average


def _component_row(modality_name, source, component_name, component_raw_metrics, cancer_types):
    return {
        'modality': modality_name,
        'source': source,
        'component': component_name,
        **{
            cancer_type: component_raw_metrics.get(cancer_type, np.nan)
            for cancer_type in cancer_types
        },
    }


def _best_decomposition_for_modality(disent_metrics, modality_name, cancer_types):
    source_short = get_honeybee_modality_short_name(modality_name)
    component_prefix = f'D_{source_short}_'

    best_components = None
    best_average = -np.inf
    for component_name, component_metrics in disent_metrics.items():
        if not component_name.startswith(component_prefix) or component_name == f'D_{source_short}_all':
            continue

        target_short = component_name[len(component_prefix):]
        unique_component = f'U_{source_short}_{target_short}'
        shared_component = f'S_{source_short}_{target_short}'
        joint_component = component_name
        if unique_component not in disent_metrics or shared_component not in disent_metrics:
            continue

        average_metric = _average_metric(component_metrics, cancer_types)
        if np.isnan(average_metric):
            continue
        if average_metric > best_average:
            best_components = (unique_component, shared_component, joint_component)
            best_average = average_metric

    return best_components, best_average


def _build_best_metric_summary_rows(disent_metrics, baseline_metrics, disent_raw_metrics, baseline_raw_metrics, cancer_types):
    rows = []
    for modality_name in HONEYBEE_MODALITIES:
        baseline_raw_metrics_for_modality = baseline_raw_metrics.get(modality_name)
        if modality_name not in baseline_metrics:
            raise KeyError(
                f"Baseline table does not contain modality row {modality_name!r}. "
                f"Available rows: {sorted(baseline_metrics.keys())}"
            )
        if baseline_raw_metrics_for_modality is None:
            raise KeyError(
                f"Baseline table does not contain raw modality row {modality_name!r}. "
                f"Available rows: {sorted(baseline_raw_metrics.keys())}"
            )

        rows.append({
            'modality': modality_name,
            'source': 'baseline',
            'component': modality_name,
            **{
                cancer_type: baseline_raw_metrics_for_modality.get(cancer_type, np.nan)
                for cancer_type in cancer_types
            },
        })

        best_component, _, _ = _best_component_for_modality(
            disent_metrics,
            modality_name,
            cancer_types,
        )
        if best_component is None:
            print(f'Skipping best summary row for {modality_name}: no valid model components were found.')
            continue

        rows.append({
            'modality': modality_name,
            'source': 'model_best_component',
            'component': best_component,
            **{
                cancer_type: disent_raw_metrics[best_component].get(cancer_type, np.nan)
                for cancer_type in cancer_types
            },
        })

        best_decomposition, _ = _best_decomposition_for_modality(
            disent_metrics,
            modality_name,
            cancer_types,
        )
        if best_decomposition is None:
            print(f'Skipping best decomposition rows for {modality_name}: no valid U/S/D triplet was found.')
            continue

        for source, component_name in zip(
            ('model_best_decomposition_unique', 'model_best_decomposition_shared', 'model_best_decomposition_joint'),
            best_decomposition,
        ):
            rows.append(_component_row(
                modality_name,
                source,
                component_name,
                disent_raw_metrics[component_name],
                cancer_types,
            ))

    return rows


def _save_best_metric_summary_table(rows, cancer_types, output_path):
    columns = ['modality', 'source', 'component', *cancer_types]
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                column: _format_table_cell(row[column]) if column in cancer_types else row[column]
                for column in columns
            })


def _save_best_component_summary_for_table(disent_metrics, baseline_metrics, disent_raw_metrics, baseline_raw_metrics, cancer_types, table_key, output_dir):
    if table_key == 'survival_analysis_component_summary':
        output_name = 'best_survival_component_summary.csv'
    elif table_key == 'cancer_type_component_summary':
        output_name = 'best_cancer_type_component_summary.csv'
    else:
        return

    summary_rows = _build_best_metric_summary_rows(
        disent_metrics,
        baseline_metrics,
        disent_raw_metrics,
        baseline_raw_metrics,
        cancer_types,
    )
    summary_path = os.path.join(output_dir, output_name)
    _save_best_metric_summary_table(summary_rows, cancer_types, summary_path)
    print(f'Saved {summary_path}')


def _plot_delta_heatmap(delta, component_rows, cancer_types, modality_name, output_path, metric_label='metric', table_key=None):
    if delta.size == 0:
        return

    fig_width = max(7.0, 1.2 * len(cancer_types) + 4.0)
    fig_height = max(5.0, 0.4 * len(component_rows) + 2.0)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        delta,
        ax=ax,
        cmap='RdBu_r',
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        xticklabels=cancer_types,
        yticklabels=component_rows,
        annot=True,
        fmt='.3f',
        linewidths=0.4,
        linecolor='white',
        cbar_kws={'label': f'Delta {metric_label} vs baseline'},
    )
    title_prefix = f'{table_key}: ' if table_key is not None else ''
    ax.set_title(f'{title_prefix}{modality_name}: disentanglement gain over simple baseline')
    ax.set_xlabel('Cancer type')
    ax.set_ylabel('Component')
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Compare Honeybee cancer-type tables from two W&B runs and plot modality-specific delta heatmaps.')
    parser.add_argument('--entity', type=str, default='vasiliki-rizou-epfl', help='W&B entity for both runs')
    parser.add_argument('--project', type=str, default='honeybee-posthoc', help='W&B project for both runs')
    parser.add_argument('--disent-run-name', type=str, default='79n0aozu', help='W&B run name or id for the disentanglement model')
    parser.add_argument('--baseline-run-name', type=str, default='6yac6zpm', help='W&B run name or id for the simple baseline model')
    parser.add_argument('--disent-table-json', type=str, nargs='+', default=None, help='Optional local path(s) to disentanglement table JSONs, one per table key')
    parser.add_argument('--baseline-table-json', type=str, nargs='+', default=None, help='Optional local path(s) to simple baseline table JSONs, one per table key')
    parser.add_argument(
        '--table-key',
        '--table-keys',
        type=str,
        nargs='+',
        default=['cancer_type_component_summary'],
        dest='table_keys',
        help='One or more W&B summary keys to compare, e.g. cancer_type_component_summary survival_analysis_component_summary',
    )
    parser.add_argument('--filter_cancer_types', nargs='+', default=DEFAULT_FILTER_CANCER_TYPES, help='Optional cancer type columns to compare, e.g. --filter_cancer_types TCGA-BRCA TCGA-LUAD or TCGA-BRCA,TCGA-LUAD. Should match training.')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    table_keys = _normalize_cli_values(args.table_keys)
    filter_cancer_types = _parse_filter_cancer_types(args.filter_cancer_types)
    for table_idx, table_key in enumerate(table_keys):
        disent_table_json = _resolve_table_json_path(
            args.disent_table_json,
            table_keys,
            table_idx,
            'disentanglement',
        )
        baseline_table_json = _resolve_table_json_path(
            args.baseline_table_json,
            table_keys,
            table_idx,
            'baseline',
        )

        disent_columns, disent_rows = _load_table(
            table_json_path=disent_table_json,
            entity=args.entity,
            project=args.project,
            run_name=args.disent_run_name,
            table_key=table_key,
        )
        baseline_columns, baseline_rows = _load_table(
            table_json_path=baseline_table_json,
            entity=args.entity,
            project=args.project,
            run_name=args.baseline_run_name,
            table_key=table_key,
        )

        disent_metrics, disent_raw_metrics, disent_cancer_types = _table_to_metric_maps(disent_columns, disent_rows)
        baseline_metrics, baseline_raw_metrics, baseline_cancer_types = _table_to_metric_maps(baseline_columns, baseline_rows)
        involved_cancer_types = _resolve_selected_cancer_types(
            disent_cancer_types,
            baseline_cancer_types,
            requested_cancer_types=filter_cancer_types,
        )

        table_output_dir = os.path.join(script_dir, 'figures', _sanitize_name(table_key))
        os.makedirs(table_output_dir, exist_ok=True)

        _save_best_component_summary_for_table(
            disent_metrics,
            baseline_metrics,
            disent_raw_metrics,
            baseline_raw_metrics,
            involved_cancer_types,
            table_key,
            table_output_dir,
        )

        metric_label = _metric_label_for_table(table_key)
        for modality_name in HONEYBEE_MODALITIES:
            component_rows, delta = _compute_delta_matrix(disent_metrics, baseline_metrics, modality_name, involved_cancer_types)
            if not component_rows:
                print(f'Skipping {table_key} / {modality_name}: no disentanglement components start with this modality.')
                continue

            out_path = os.path.join(table_output_dir, f'delta_heatmap_{_sanitize_name(modality_name)}.pdf')
            _plot_delta_heatmap(
                delta,
                component_rows,
                involved_cancer_types,
                modality_name,
                out_path,
                metric_label=metric_label,
                table_key=table_key if len(table_keys) > 1 else None,
            )
            print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
