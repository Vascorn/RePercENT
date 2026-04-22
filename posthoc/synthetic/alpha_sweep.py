import argparse
import math
import os
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from posthoc.plotting_config import apply_paper_plot_style

apply_paper_plot_style()

ENTITY = "vasiliki-rizou-epfl"

PROJECT = "repercent_alpha_ablation_synthetic"
BASE_PROJECTS = [
    "posthoc_synthetic-2M-final",
    "posthoc_synthetic-3M-final",
    "posthoc_synthetic-4M-final",
    "posthoc_synthetic-5M-final"
]

ACC_METRICS = {
    "u2u_acc": ("u2u_acc_mean", "u2u_acc_std"),
    "u2s_acc": ("u2s_acc_mean", "u2s_acc_std"),
    "s2s_acc": ("s2s_acc_mean", "s2s_acc_std"),
    "s2u_acc": ("s2u_acc_mean", "s2u_acc_std"),
}

METRIC_TITLES = {
    "u2u_acc": r"Accuracy $\mathrm{u} \to \mathrm{y_u}$",
    "u2s_acc": r"Accuracy $\mathrm{u} \to \mathrm{y_s}$",
    "s2s_acc": r"Accuracy $\mathrm{s} \to \mathrm{y_s}$",
    "s2u_acc": r"Accuracy $\mathrm{s} \to \mathrm{y_u}$",
    "delta_u": r"$\Delta_\mathrm{u} = (\mathrm{u} \to \mathrm{y_u}) - (\mathrm{u} \to \mathrm{y_s})$",
    "delta_s": r"$\Delta_\mathrm{s} = (\mathrm{s} \to \mathrm{y_s}) - (\mathrm{s} \to \mathrm{y_u})$",
}

DELTA_METRICS = {
    "delta_u": ("u2u_acc", "u2s_acc"),
    "delta_s": ("s2s_acc", "s2u_acc"),
}


MODALITY_MARKERS = {
    "2": "o",
    "3": "s",
    "4": "^",
    "5": "D"
}

MODALITY_COLORS = {
    "2": "#984EA3",
    "3": "#377EB8",
    "4": "#E41A1C",
    "5": "#4DAF4A",
}


NEUTRAL_PLOT_COLOR = "#6E6E6E"

MODALITY_LINESTYLES = {
    "2": "-",
    "3": "--",
    "4": "-.",
    "5": ":"
}

TITLE_FONTSIZE = 16
AXIS_LABEL_FONTSIZE = 18
TICK_LABEL_FONTSIZE = 10
ANNOTATION_FONTSIZE = 10
LEGEND_FONTSIZE = 10
LEGEND_TITLE_FONTSIZE = 10
COLORBAR_LABEL_FONTSIZE = 15
COLORBAR_TICK_FONTSIZE = 13
FIGSIZE = (7, 5)


def _run_name(num_modalities: str, alpha: str) -> str:
    return f"repercent_splits3_{num_modalities}M_M{num_modalities}_alpha{alpha}"


def _candidate_run_names(num_modalities: str, alpha: str) -> list[str]:
    return [
        _run_name(num_modalities, alpha),
        f"repercent_splits_3_{num_modalities}M_M{num_modalities}_alpha{alpha}",
    ]


def _table_key(num_modalities: str, alpha: str) -> str:
    return f"final_metrics/M{num_modalities}/alpha{alpha}"


def _artifact_matches_table_key(artifact, table_key: str) -> bool:
    escaped_table_key = table_key.replace(".", r"\.")
    compact_table_key = re.sub(r"[^A-Za-z0-9_]+", "", table_key)
    artifact_name = str(getattr(artifact, "name", ""))
    aliases = [str(alias) for alias in getattr(artifact, "aliases", [])]
    return (
        table_key in artifact_name
        or escaped_table_key in artifact_name
        or compact_table_key in artifact_name
        or any(table_key in alias or escaped_table_key in alias or compact_table_key in alias for alias in aliases)
    )


def _download_metrics_table(run, table_key: str) -> pd.DataFrame:
    artifacts = list(run.logged_artifacts())
    for artifact in artifacts:
        if _artifact_matches_table_key(artifact, table_key):
            for candidate_key in (table_key, table_key.replace(".", r"\.")):
                try:
                    return artifact.get(candidate_key).get_dataframe()
                except KeyError:
                    pass

    if table_key == "final_metrics/all_runs":
        for artifact in artifacts:
            if "final_metrics" not in str(getattr(artifact, "name", "")):
                continue
            try:
                return artifact.get(table_key).get_dataframe()
            except KeyError:
                pass

    available = [str(getattr(artifact, "name", "")) for artifact in artifacts]
    raise KeyError(
        f"Run {run.path} does not have a logged table for {table_key!r}. "
        f"Available artifacts: {available}"
    )


def _metrics_from_table(table_df: pd.DataFrame) -> dict[str, float]:
    if not {"metric", "value"}.issubset(table_df.columns):
        raise ValueError(f"Expected table columns ['metric', 'value'], got {table_df.columns.tolist()}")

    metric_values = table_df.set_index("metric")["value"].to_dict()
    extracted = {}
    for metric_name, (mean_key, std_key) in ACC_METRICS.items():
        extracted[f"{metric_name}_mean"] = _lookup_metric_value(metric_values, mean_key)
        extracted[f"{metric_name}_std"] = _lookup_metric_value(metric_values, std_key)
    return extracted


def _lookup_metric_value(metric_values: dict[str, float], metric_key: str) -> float | None:
    if metric_key in metric_values:
        return metric_values[metric_key]

    suffix_matches = [value for key, value in metric_values.items() if str(key).endswith(f"/{metric_key}")]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    return None


def _extract_num_modalities(project_name: str) -> int | None:
    match = re.search(r"(?:^|[-_])(\d+)M(?:[-_]|$)", str(project_name), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _normalize_modalities(modalities: list[str]) -> list[str]:
    normalized = []
    for modality in modalities:
        for token in str(modality).split(","):
            token = token.strip()
            if not token:
                continue
            normalized.append(str(int(token)))
    return normalized


def _normalize_alpha_values(alpha_values: list[str]) -> list[str]:
    normalized = []
    for alpha in alpha_values:
        for token in str(alpha).split(","):
            token = token.strip()
            if not token:
                continue
            normalized.append(token)
    return normalized


def _metrics_from_wide_row(row: pd.Series) -> dict[str, float]:
    row_values = row.to_dict()
    extracted = {}
    for metric_name, (mean_key, std_key) in ACC_METRICS.items():
        extracted[f"{metric_name}_mean"] = _lookup_metric_value(row_values, mean_key)
        extracted[f"{metric_name}_std"] = _lookup_metric_value(row_values, std_key)
    return extracted


def _fetch_alpha_metrics(api: wandb.Api, entity: str, project: str, modalities: list[str], alphas: list[str]) -> pd.DataFrame:
    rows = []
    for num_modalities in modalities:
        for alpha in alphas:
            table_key = _table_key(num_modalities, alpha)
            run_names = _candidate_run_names(num_modalities, alpha)
            runs = []
            for run_name in run_names:
                runs = api.runs(f"{entity}/{project}", filters={"display_name": run_name})
                if len(runs) > 0:
                    break
            if len(runs) == 0:
                print(f"Missing run: {', '.join(run_names)}")
                continue
            if len(runs) > 1:
                print(f"Found {len(runs)} runs named {run_name}; using the first one.")

            run = runs[0]
            table_df = _download_metrics_table(run, table_key)
            rows.append(
                {
                    "num_modalities": int(num_modalities),
                    "alpha": float(alpha),
                    "alpha_label": alpha,
                    "run_name": run.name,
                    "run_id": run.id,
                    "source": "alpha_ablation",
                    **_metrics_from_table(table_df),
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["num_modalities", "alpha"])


def _fetch_base_alpha_metrics(
    api: wandb.Api,
    entity: str,
    projects: list[str],
    modalities: list[str],
    alpha: float,
    run_name: str,
) -> pd.DataFrame:
    requested_modalities = {int(modality) for modality in _normalize_modalities(modalities)}
    rows = []
    for project in projects:
        num_modalities = _extract_num_modalities(project)
        if num_modalities is None or num_modalities not in requested_modalities:
            continue

        runs = api.runs(f"{entity}/{project}", filters={"display_name": run_name})
        if len(runs) == 0:
            print(f"Missing base run: {entity}/{project}/{run_name}")
            continue
        if len(runs) > 1:
            print(f"Found {len(runs)} base runs named {run_name} in {project}; using the first one.")

        run = runs[0]
        table_df = _download_metrics_table(run, "final_metrics/all_runs")
        selected = table_df[
            (pd.to_numeric(table_df["split_idx"], errors="coerce") == 1)
            & (pd.to_numeric(table_df["seed_idx"], errors="coerce") == 0)
        ]
        if selected.empty:
            print(f"Missing split_idx=1, seed_idx=0 row in {entity}/{project}/{run_name}")
            continue

        row = selected.iloc[0]
        rows.append(
            {
                "num_modalities": num_modalities,
                "alpha": float(alpha),
                "alpha_label": f"{alpha:.1f}",
                "run_name": run.name,
                "run_id": run.id,
                "source": "base_project",
                **_metrics_from_wide_row(row),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["num_modalities", "alpha"])


def _metric_percent_multiplier(df: pd.DataFrame, columns: list[str]) -> float:
    values = pd.concat([df[column] for column in columns], ignore_index=True).dropna()
    if values.empty:
        return 1.0
    return 1.0 if values.abs().max() > 1.0 else 100.0


def _plot_metric_alpha_sweep(df: pd.DataFrame, out_dir: str, metric_name: str) -> None:
    plot_df = df.copy()
    mean_plot_col = f"{metric_name}_mean"
    std_plot_col = f"{metric_name}_std"
    multiplier = _metric_percent_multiplier(plot_df, [mean_plot_col, std_plot_col])
    plot_df[mean_plot_col] = plot_df[mean_plot_col] * multiplier
    plot_df[std_plot_col] = plot_df[std_plot_col] * multiplier

    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)

    for num_modalities, modality_df in plot_df.groupby("num_modalities"):
        modality = str(num_modalities)
        modality_df = modality_df.sort_values("alpha")
        color = MODALITY_COLORS[modality]
        ax.plot(
            modality_df["alpha"],
            modality_df[mean_plot_col],
            color=color,
            linestyle=MODALITY_LINESTYLES[modality],
            marker=MODALITY_MARKERS[modality],
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=0.8,
            linewidth=3.0,
            alpha=0.9,
            label=f"{modality} M",
        )
        if modality_df[std_plot_col].notna().any():
            ax.fill_between(
                modality_df["alpha"],
                modality_df[mean_plot_col] - modality_df[std_plot_col],
                modality_df[mean_plot_col] + modality_df[std_plot_col],
                color=color,
                alpha=0.10,
                linewidth=0,
            )

    alpha_ticks = sorted(plot_df["alpha"].unique())
    alpha_labels = [
        plot_df.loc[plot_df["alpha"] == alpha, "alpha_label"].iloc[0]
        for alpha in alpha_ticks
    ]
    ax.set_xscale("log")
    ax.set_xticks(alpha_ticks)
    ax.set_xticklabels(alpha_labels)
    ax.set_title(METRIC_TITLES[metric_name], fontsize=TITLE_FONTSIZE, pad=10)
    ax.set_xlabel(r"$\alpha$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Accuracy (%)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylim(50 if metric_name in ["u2s_acc", "s2u_acc"] else 60, 80 if metric_name in ["u2s_acc", "s2u_acc"] else 95)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.18, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="best")

    out_path = os.path.join(out_dir, f"{metric_name}_alpha_sweep.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def _plot_delta_alpha_sweep(df: pd.DataFrame, out_dir: str, delta_name: str) -> None:
    positive_metric, negative_metric = DELTA_METRICS[delta_name]
    positive_mean = f"{positive_metric}_mean"
    positive_std = f"{positive_metric}_std"
    negative_mean = f"{negative_metric}_mean"
    negative_std = f"{negative_metric}_std"

    plot_df = df.copy()
    multiplier = _metric_percent_multiplier(plot_df, [positive_mean, positive_std, negative_mean, negative_std])
    plot_df[f"{delta_name}_mean"] = (plot_df[positive_mean] - plot_df[negative_mean]) * multiplier
    plot_df[f"{delta_name}_std"] = (
        (plot_df[positive_std] * multiplier) ** 2 + (plot_df[negative_std] * multiplier) ** 2
    ).apply(math.sqrt)

    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)

    for num_modalities, modality_df in plot_df.groupby("num_modalities"):
        modality = str(num_modalities)
        modality_df = modality_df.sort_values("alpha")
        color = MODALITY_COLORS[modality]
        ax.plot(
            modality_df["alpha"],
            modality_df[f"{delta_name}_mean"],
            color=color,
            linestyle=MODALITY_LINESTYLES[modality],
            marker=MODALITY_MARKERS[modality],
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=0.8,
            linewidth=3.0,
            alpha=0.9,
            label=f"{modality} M",
        )
        if modality_df[f"{delta_name}_std"].notna().any():
            ax.fill_between(
                modality_df["alpha"],
                modality_df[f"{delta_name}_mean"] - modality_df[f"{delta_name}_std"],
                modality_df[f"{delta_name}_mean"] + modality_df[f"{delta_name}_std"],
                color=color,
                alpha=0.10,
                linewidth=0,
            )

    alpha_ticks = sorted(plot_df["alpha"].unique())
    alpha_labels = [
        plot_df.loc[plot_df["alpha"] == alpha, "alpha_label"].iloc[0]
        for alpha in alpha_ticks
    ]
    y_max = max(5.0, plot_df[f"{delta_name}_mean"].max() + plot_df[f"{delta_name}_std"].fillna(0).max())
    ax.set_xscale("log")
    ax.set_xticks(alpha_ticks)
    ax.set_xticklabels(alpha_labels)
    ax.set_title(METRIC_TITLES[delta_name], fontsize=TITLE_FONTSIZE, pad=10)
    ax.set_xlabel(r"$\alpha$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Accuracy margin (%)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylim(0, y_max * 1.12)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.18, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="best")

    out_path = os.path.join(out_dir, f"{delta_name}_alpha_sweep.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}")




def main():
    parser = argparse.ArgumentParser(description="Plot alpha ablations from wandb final metric tables")
    parser.add_argument("--entity", type=str, default=ENTITY, help="Wandb entity name")
    parser.add_argument("--project", type=str, default=PROJECT, help="Wandb project name")
    parser.add_argument("--base_projects", nargs="+", default=BASE_PROJECTS, help="Wandb core experiment projects used for the alpha=2.0 point")
    parser.add_argument("--base_run_name", type=str, default="aggregate_repercent", help="Run name to read from each core experiment project")
    parser.add_argument("--base_alpha", type=float, default=2.0, help="Alpha value assigned to the core experiment point")
    parser.add_argument("--out_dir", type=str, default=os.path.join(os.path.dirname(__file__), "figures", "alpha_ablations"), help="Directory to save plots")
    parser.add_argument("--modalities", nargs="+", default=["3", "4", "5"], help="List of modality counts to consider for plotting")
    parser.add_argument("--alpha_values", nargs="+", default=["0.01", "0.1", "1.0", "2.0", "10.0", "100.0"], help="List of alpha values to consider for plotting")

    args = parser.parse_args()
    args.modalities = _normalize_modalities(args.modalities)
    args.alpha_values = _normalize_alpha_values(args.alpha_values)

    api = wandb.Api()
    alpha_values = [str(alpha) for alpha in args.alpha_values]
    base_alpha_requested = any(math.isclose(float(alpha), args.base_alpha) for alpha in alpha_values)
    ablation_alpha_values = [
        alpha for alpha in alpha_values
        if not math.isclose(float(alpha), args.base_alpha)
    ]

    frames = []
    if ablation_alpha_values:
        frames.append(_fetch_alpha_metrics(api, args.entity, args.project, args.modalities, ablation_alpha_values))
    if base_alpha_requested:
        frames.append(
            _fetch_base_alpha_metrics(
                api=api,
                entity=args.entity,
                projects=args.base_projects,
                modalities=args.modalities,
                alpha=args.base_alpha,
                run_name=args.base_run_name,
            )
        )

    df = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if frames else pd.DataFrame()
    if not df.empty:
        df = df.sort_values(["num_modalities", "alpha"])
    if df.empty:
        raise RuntimeError("No alpha ablation metrics were found.")

    metric_columns = [
        "u2u_acc_mean",
        "u2u_acc_std",
        "u2s_acc_mean",
        "u2s_acc_std",
        "s2s_acc_mean",
        "s2s_acc_std",
        "s2u_acc_mean",
        "s2u_acc_std",
    ]
    print(df[["num_modalities", "alpha_label", *metric_columns]].to_string(index=False))

    for metric_name in ACC_METRICS:
        _plot_metric_alpha_sweep(df, args.out_dir, metric_name)
    for delta_name in DELTA_METRICS:
        _plot_delta_alpha_sweep(df, args.out_dir, delta_name)


if __name__ == "__main__":
    main()
