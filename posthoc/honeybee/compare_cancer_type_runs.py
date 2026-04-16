import argparse
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

from posthoc.honeybee.helper_metrics import HONEYBEE_MODALITIES, get_honeybee_modality_short_name


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


def _table_to_metric_map(columns, rows):
    if 'component' not in columns:
        raise ValueError(f"Expected a 'component' column, but got columns: {columns}")

    component_idx = columns.index('component')
    metric_columns = [column for column in columns if column not in {'component', 'overall'}]
    metric_indices = {column: columns.index(column) for column in metric_columns}

    metrics = {}
    for row in rows:
        component_name = str(row[component_idx])
        metrics[component_name] = {
            cancer_type: _extract_float(row[idx])
            for cancer_type, idx in metric_indices.items()
        }

    return metrics, metric_columns


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


def _plot_delta_heatmap(delta, component_rows, cancer_types, modality_name, output_path):
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
        cbar_kws={'label': 'Delta accuracy vs baseline'},
    )
    ax.set_title(f'{modality_name}: disentanglement gain over simple baseline')
    ax.set_xlabel('Cancer type')
    ax.set_ylabel('Component')
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Compare Honeybee cancer-type tables from two W&B runs and plot modality-specific delta heatmaps.')
    parser.add_argument('--entity', type=str, default='vasiliki-rizou-epfl', help='W&B entity for both runs')
    parser.add_argument('--project', type=str, default='honeybee-posthoc', help='W&B project for both runs')
    parser.add_argument('--disent-run-name', type=str, default='8yudodul', help='W&B run name or id for the disentanglement model')
    parser.add_argument('--baseline-run-name', type=str, default='2g0smsdp', help='W&B run name or id for the simple baseline model')
    parser.add_argument('--disent-table-json', type=str, default=None, help='Optional local path to the disentanglement cancer_type_component_summary table JSON')
    parser.add_argument('--baseline-table-json', type=str, default=None, help='Optional local path to the simple baseline cancer_type_component_summary table JSON')
    parser.add_argument('--table-key', type=str, default='cancer_type_component_summary', help='W&B summary key for both classification tables')
    parser.add_argument('--output-dir', type=str, default='figures/cancer_type_delta_heatmaps', help='Output directory relative to this script')
    args = parser.parse_args()


    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    disent_columns, disent_rows = _load_table(
        table_json_path=args.disent_table_json,
        entity=args.entity,
        project=args.project,
        run_name=args.disent_run_name,
        table_key=args.table_key,
    )
    baseline_columns, baseline_rows = _load_table(
        table_json_path=args.baseline_table_json,
        entity=args.entity,
        project=args.project,
        run_name=args.baseline_run_name,
        table_key=args.table_key,
    )

    disent_metrics, disent_cancer_types = _table_to_metric_map(disent_columns, disent_rows)
    baseline_metrics, baseline_cancer_types = _table_to_metric_map(baseline_columns, baseline_rows)

    cancer_types = [cancer_type for cancer_type in disent_cancer_types if cancer_type in baseline_cancer_types]
    if not cancer_types:
        raise ValueError('No shared cancer type columns were found between the two tables.')

    for modality_name in HONEYBEE_MODALITIES:
        component_rows, delta = _compute_delta_matrix(disent_metrics, baseline_metrics, modality_name, cancer_types)
        if not component_rows:
            print(f'Skipping {modality_name}: no disentanglement components start with this modality.')
            continue

        out_path = os.path.join(output_dir, f'delta_heatmap_{_sanitize_name(modality_name)}.pdf')
        _plot_delta_heatmap(delta, component_rows, cancer_types, modality_name, out_path)
        print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
