import argparse
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

PROJECTS = [
    "posthoc_synthetic-2M-final",
    "posthoc_synthetic-3M-final",
    "posthoc_synthetic-4M-final",
    "posthoc_synthetic-5M-final",
]

METRICS = [
    "model_params",
    "final/linear_probe/u2u_acc_mean_mean",
    "final/linear_probe/u2u_acc_mean_std",
    "final/linear_probe/u2s_acc_mean_mean",
    "final/linear_probe/u2s_acc_mean_std",
    "final/linear_probe/s2s_acc_mean_mean",
    "final/linear_probe/s2s_acc_mean_std",
    "final/linear_probe/s2u_acc_mean_mean",
    "final/linear_probe/s2u_acc_mean_std",
    "final/linear_probe/u2u_recall_mean_mean",
    "final/linear_probe/u2u_recall_mean_std",
    "final/linear_probe/u2s_recall_mean_mean",
    "final/linear_probe/u2s_recall_mean_std",
    "final/linear_probe/s2s_recall_mean_mean",
    "final/linear_probe/s2s_recall_mean_std",
    "final/linear_probe/s2u_recall_mean_mean",
    "final/linear_probe/s2u_recall_mean_std",
    "final/linear_probe/u2u_f1_mean_mean",
    "final/linear_probe/u2u_f1_mean_std",
    "final/linear_probe/u2s_f1_mean_mean",
    "final/linear_probe/u2s_f1_mean_std",
    "final/linear_probe/s2s_f1_mean_mean",
    "final/linear_probe/s2s_f1_mean_std",
    "final/linear_probe/s2u_f1_mean_mean",
    "final/linear_probe/s2u_f1_mean_std",
    "final/linear_probe/u2u_mcc_mean_mean",
    "final/linear_probe/u2u_mcc_mean_std",
    "final/linear_probe/u2s_mcc_mean_mean",
    "final/linear_probe/u2s_mcc_mean_std",
    "final/linear_probe/s2s_mcc_mean_mean",
    "final/linear_probe/s2s_mcc_mean_std",
    "final/linear_probe/s2u_mcc_mean_mean",
    "final/linear_probe/s2u_mcc_mean_std",
]

ACC_METRICS = {
    "u2u_acc": ("final/linear_probe/u2u_acc_mean_mean", "final/linear_probe/u2u_acc_mean_std"),
    "u2s_acc": ("final/linear_probe/u2s_acc_mean_mean", "final/linear_probe/u2s_acc_mean_std"),
    "s2s_acc": ("final/linear_probe/s2s_acc_mean_mean", "final/linear_probe/s2s_acc_mean_std"),
    "s2u_acc": ("final/linear_probe/s2u_acc_mean_mean", "final/linear_probe/s2u_acc_mean_std"),
}

METRIC_TITLES = {
    "u2u_acc": r"Accuracy $\mathrm{u} \to \mathrm{u}$",
    "u2s_acc": r"Accuracy $\mathrm{u} \to \mathrm{s}$",
    "s2s_acc": r"Accuracy $\mathrm{s} \to \mathrm{s}$",
    "s2u_acc": r"Accuracy $\mathrm{s} \to \mathrm{u}$",
    "delta_to_ideal": r"$\Delta_{\mathrm{model}}$",
}


MODEL_MARKERS = {
    "RePercENT": "o",
    "gMLP": "s",
    "GRU": "^",
    "MLP": "D",
    "JointOpt": "P",
    "Other": "X",
}

MODEL_COLORS = {
    "RePercENT": "#E41A1C",
    "gMLP": "#377EB8",
    "GRU": "#4DAF4A",
    "MLP": "#984EA3",
    "JointOpt": "#6E6E6E",
    "Other": "#6E6E6E",
}

MODEL_ORDER = ["RePercENT", "gMLP", "GRU", "MLP", "JointOpt", "Other"]
NEUTRAL_PLOT_COLOR = "#6E6E6E"

MODEL_LINESTYLES = {
    "RePercENT": "-",
    "gMLP": "--",
    "GRU": "-.",
    "MLP": ":",
    "JointOpt": (0, (5, 1.5)),
    "Other": (0, (2, 1.2)),
}

TITLE_FONTSIZE = 22
AXIS_LABEL_FONTSIZE = 22
TICK_LABEL_FONTSIZE = 15
ANNOTATION_FONTSIZE = 12
LEGEND_FONTSIZE = 15
LEGEND_TITLE_FONTSIZE = 15
COLORBAR_LABEL_FONTSIZE = 20
COLORBAR_TICK_FONTSIZE = 17
COMPLEXITY_FIGSIZE = (7, 5)


def _metric_average_color(family_df: pd.DataFrame, cmap, vmin: float | None, vmax: float | None):
    average_metric = family_df["metric_mean"].mean()
    if pd.isna(average_metric):
        return NEUTRAL_PLOT_COLOR
    if vmin is None or vmax is None:
        vmin = family_df["metric_mean"].min()
        vmax = family_df["metric_mean"].max()
    if pd.isna(vmin) or pd.isna(vmax) or vmax <= vmin:
        return cmap(0.5)
    normalized_metric = min(max((average_metric - vmin) / (vmax - vmin), 0.0), 1.0)
    return cmap(normalized_metric)


def _extract_num_modalities(project_name: str) -> int | None:
    match = re.search(r"(?:^|[-_])(\d+)M(?:[-_]|$)", str(project_name), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _extract_model_family(run_name: str) -> str:
    name = str(run_name).strip().lower()
    if "repercent" in name:
        return "RePercENT"
    if "gmlp" in name:
        return "gMLP"
    if "gru" in name:
        return "GRU"
    if re.search(r"(^|[_\-\s])mlp($|[_\-\s])", name):
        return "MLP"
    if "jointopt" in name:
        return "JointOpt"
    return "Other"


def _aggregate_plot_df(df: pd.DataFrame, mean_col: str, std_col: str) -> pd.DataFrame:
    plot_df = df[["project", "run_name", "model_params", mean_col, std_col]].copy()
    plot_df = plot_df.dropna(subset=["model_params", mean_col])
    plot_df["num_modalities"] = plot_df["project"].map(_extract_num_modalities)
    plot_df["model_family"] = plot_df["run_name"].map(_extract_model_family)
    plot_df = plot_df.dropna(subset=["num_modalities"])
    plot_df["num_modalities"] = plot_df["num_modalities"].astype(int)

    grouped = (
        plot_df.groupby(["model_family", "num_modalities"], as_index=False)
        .agg(
            model_params=("model_params", "mean"),
            metric_mean=(mean_col, "mean"),
            metric_std=(std_col, "mean"),
            n_runs=("run_name", "count"),
        )
        .sort_values(["num_modalities", "model_params", "model_family"])
    )
    return grouped


def _metric_percent_multiplier(series: pd.Series) -> float:
    non_null = series.dropna()
    if non_null.empty:
        return 1.0
    return 1.0 if non_null.abs().max() > 1.0 else 100.0


def _aggregate_delta_to_ideal_plot_df(df: pd.DataFrame) -> pd.DataFrame:
    u2u_mean_col, u2u_std_col = ACC_METRICS["u2u_acc"]
    u2s_mean_col, u2s_std_col = ACC_METRICS["u2s_acc"]
    s2s_mean_col, s2s_std_col = ACC_METRICS["s2s_acc"]
    s2u_mean_col, s2u_std_col = ACC_METRICS["s2u_acc"]
    mean_cols = [u2u_mean_col, u2s_mean_col, s2s_mean_col, s2u_mean_col]
    std_cols = [u2u_std_col, u2s_std_col, s2s_std_col, s2u_std_col]

    plot_df = df[["project", "run_name", "model_params", *mean_cols, *std_cols]].copy()
    plot_df = plot_df.dropna(subset=["model_params", *mean_cols])

    percent_multipliers = {mean_col: _metric_percent_multiplier(plot_df[mean_col]) for mean_col in mean_cols}
    u2u = plot_df[u2u_mean_col] * percent_multipliers[u2u_mean_col]
    u2s = plot_df[u2s_mean_col] * percent_multipliers[u2s_mean_col]
    s2s = plot_df[s2s_mean_col] * percent_multipliers[s2s_mean_col]
    s2u = plot_df[s2u_mean_col] * percent_multipliers[s2u_mean_col]

    plot_df["metric_mean"] = ((100.0 - u2u).abs() + (u2s - 50.0).abs() + (100.0 - s2s).abs() + (s2u - 50.0).abs()) / 4.0
    plot_df["metric_std"] = plot_df.apply(
        lambda row: (
            (
                (row[u2u_std_col] * percent_multipliers[u2u_mean_col]) ** 2
                + (row[u2s_std_col] * percent_multipliers[u2s_mean_col]) ** 2
                + (row[s2s_std_col] * percent_multipliers[s2s_mean_col]) ** 2
                + (row[s2u_std_col] * percent_multipliers[s2u_mean_col]) ** 2
            )
            ** 0.5
        ) / 4.0
        if all(pd.notna(row[col]) for col in std_cols)
        else None,
        axis=1,
    )
    plot_df["num_modalities"] = plot_df["project"].map(_extract_num_modalities)
    plot_df["model_family"] = plot_df["run_name"].map(_extract_model_family)
    plot_df = plot_df.dropna(subset=["num_modalities"])
    plot_df["num_modalities"] = plot_df["num_modalities"].astype(int)

    grouped = (
        plot_df.groupby(["model_family", "num_modalities"], as_index=False)
        .agg(
            model_params=("model_params", "mean"),
            metric_mean=("metric_mean", "mean"),
            metric_std=("metric_std", "mean"),
            n_runs=("run_name", "count"),
        )
        .sort_values(["num_modalities", "model_params", "model_family"])
    )
    return grouped


def _plot_grouped_metric(
    grouped: pd.DataFrame,
    out_dir: str,
    metric_name: str,
    vmin: float | None,
    vmax: float | None,
    cmap_name: str,
    colorbar_label: str,
    annotation_suffix: str,
    annotation_format: str = ".1f",
    use_heatmap: bool = True,
    connect_points: bool = True,
    use_model_colors: bool = True,
    line_color_mode: str = "model",
    use_model_linestyles: bool = False,
) -> None:
    if grouped.empty:
        print(f"No rows available for plotting {metric_name}.")
        return

    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=COMPLEXITY_FIGSIZE, constrained_layout=True)

    cmap = plt.get_cmap(cmap_name)
    label_offsets = {
        "RePercENT": (-10, 10),
        "gMLP": (-10, 10),
        "GRU": (10, 1),
        "MLP": (-13, 12),
        "JointOpt": (12, 11),
        "Other": (-18, -11),
    }
    scatter_ref = None
    modality_ticks = sorted(grouped["num_modalities"].unique())

    for family_name, family_df in grouped.groupby("model_family"):
        family_df = family_df.sort_values("num_modalities")
        marker = MODEL_MARKERS.get(family_name, MODEL_MARKERS["Other"])
        family_color = MODEL_COLORS.get(family_name, MODEL_COLORS["Other"])
        if line_color_mode == "metric_average":
            plot_color = _metric_average_color(family_df, cmap, vmin, vmax)
        elif use_model_colors:
            plot_color = family_color
        else:
            plot_color = NEUTRAL_PLOT_COLOR
        line_style = MODEL_LINESTYLES.get(family_name, MODEL_LINESTYLES["Other"]) if use_model_linestyles else "-"
        is_repercent = family_name == "RePercENT"
        if connect_points and len(family_df) > 1:
            ax.plot(
                family_df["num_modalities"],
                family_df["model_params"] / 1e6,
                color=plot_color,
                linestyle=line_style,
                linewidth=3.0,
                alpha=0.9,
                zorder=2,
            )

        scatter_kwargs = {
            "marker": marker,
            "s": 120,
            "linewidths": 3.0,
            "alpha": 1.0,
            "zorder": 4,
        }
        if use_heatmap:
            scatter_ref = ax.scatter(
                family_df["num_modalities"],
                family_df["model_params"] / 1e6,
                c=family_df["metric_mean"],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                edgecolors=plot_color,
                **scatter_kwargs,
            )
        else:
            scatter_ref = ax.scatter(
                family_df["num_modalities"],
                family_df["model_params"] / 1e6,
                color=plot_color,
                edgecolors="black",
                **scatter_kwargs,
            )

        dx, dy = label_offsets.get(family_name, (6, 6))
        for _, row in family_df.iterrows():
            if row["num_modalities"] <= 3:
                continue
            ax.annotate(
                f"{row['metric_mean']:{annotation_format}}{annotation_suffix}",
                (row["num_modalities"], row["model_params"] / 1e6),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=ANNOTATION_FONTSIZE,
                color=plot_color if use_model_colors and not use_heatmap else "black",
                alpha=1.0,
                weight="semibold",
            )

    title = METRIC_TITLES.get(metric_name, metric_name.replace("_", " ").title())
    ax.set_title(f"{title} vs Complexity", fontsize=TITLE_FONTSIZE, pad=14)
    ax.set_xlabel("M", fontsize=AXIS_LABEL_FONTSIZE, labelpad=8)
    ax.set_ylabel("Parameter count (M)", fontsize=AXIS_LABEL_FONTSIZE, labelpad=8)
    ax.set_xticks(modality_ticks)
    ax.set_xlim(min(modality_ticks) - 0.35, max(modality_ticks) + 0.35)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.yaxis.offsetText.set_fontsize(TICK_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if use_heatmap and scatter_ref is not None:
        cbar = fig.colorbar(scatter_ref, ax=ax)
        cbar.set_label(colorbar_label, fontsize=COLORBAR_LABEL_FONTSIZE, labelpad=10)
        cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE, length=3)

    legend_handles = []
    legend_labels = []
    for family_name in MODEL_ORDER:
        if family_name not in grouped["model_family"].unique():
            continue
        line_style = MODEL_LINESTYLES.get(family_name, MODEL_LINESTYLES["Other"]) if use_model_linestyles else "-"
        family_color = MODEL_COLORS.get(family_name, MODEL_COLORS["Other"])
        if line_color_mode == "model" and use_model_colors:
            plot_color = family_color
            markerfacecolor = plot_color if not use_heatmap else "white"
            markeredgecolor = plot_color
        else:
            plot_color = NEUTRAL_PLOT_COLOR
            markerfacecolor = "white" if use_heatmap else plot_color
            markeredgecolor = plot_color
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=MODEL_MARKERS.get(family_name, MODEL_MARKERS["Other"]),
                linestyle=line_style if connect_points else "None",
                markerfacecolor=markerfacecolor,
                markeredgecolor=markeredgecolor,
                markeredgewidth=1.0,
                markersize=10,
                color=plot_color,
                linewidth=3.0
            )
        )
        legend_labels.append(family_name)
    legend = ax.legend(
        legend_handles,
        legend_labels,
        title="Model",
        loc="upper left",
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_TITLE_FONTSIZE,
        handlelength=3.0,
        borderpad=0.6,
        labelspacing=0.45,
    )
    for text in legend.get_texts():
        if text.get_text() == "RePercENT":
            text.set_fontweight("bold")

    suffix = "heatmap" if use_heatmap else "trajectory"
    out_path = os.path.join(out_dir, f"{metric_name}_modalities_params_{suffix}.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def _plot_metric(
    df: pd.DataFrame,
    out_dir: str,
    metric_name: str,
    mean_col: str,
    std_col: str,
    vmin: float,
    vmax: float,
    use_heatmap: bool,
    use_model_colors: bool,
) -> None:
    grouped = _aggregate_plot_df(df, mean_col, std_col)
    _plot_grouped_metric(
        grouped=grouped,
        out_dir=out_dir,
        metric_name=metric_name,
        vmin=vmin,
        vmax=vmax,
        cmap_name="PuBu",
        colorbar_label=r"$\mathbb{E}[Acc]$ (\%)",
        annotation_suffix="",
        annotation_format=".1f",
        use_heatmap=use_heatmap,
        connect_points=True,
        use_model_colors=use_model_colors,
        line_color_mode="metric_average" if use_heatmap else "neutral",
    )


def _plot_delta_to_ideal(df: pd.DataFrame, out_dir: str) -> None:
    grouped = _aggregate_delta_to_ideal_plot_df(df)
    _plot_grouped_metric(
        grouped=grouped,
        out_dir=out_dir,
        metric_name="delta_to_ideal",
        vmin=None,
        vmax=None,
        cmap_name="PuBu",
        colorbar_label="",
        annotation_suffix="",
        annotation_format=".1f",
        use_heatmap=False,
        connect_points=True,
        use_model_colors=True,
        line_color_mode="model",
        use_model_linestyles=True,
    )
    _plot_grouped_metric(
        grouped=grouped,
        out_dir=out_dir,
        metric_name="delta_to_ideal",
        vmin=10.0,
        vmax=20.0,
        cmap_name="PuBu",
        colorbar_label=r"$\Delta_{\mathrm{model}}$ (lower is better)",
        annotation_suffix="",
        annotation_format=".1f",
        use_heatmap=True,
        connect_points=True,
        use_model_colors=False,
        line_color_mode="metric_average",
        use_model_linestyles=True,
    )


def _resolve_metric_bounds(args: argparse.Namespace) -> dict[str, tuple[float, float]]:
    metric_bounds = {}
    for metric_name in ACC_METRICS:
        metric_prefix = metric_name.replace("_acc", "")
        vmin = getattr(args, f"{metric_prefix}_vmin")
        vmax = getattr(args, f"{metric_prefix}_vmax")
        vmin = args.acc_vmin if vmin is None else vmin
        vmax = args.acc_vmax if vmax is None else vmax
        if vmin >= vmax:
            raise ValueError(
                f"{metric_name} heatmap bounds are invalid: vmin={vmin} must be smaller than vmax={vmax}."
            )
        metric_bounds[metric_name] = (vmin, vmax)
    return metric_bounds


def main():
    parser = argparse.ArgumentParser(description="Aggregate metrics from multiple wandb projects")
    parser.add_argument("--entity", type=str, default=ENTITY, help="Wandb entity name")
    parser.add_argument("--projects", nargs="+", default=PROJECTS, help="List of wandb project names to aggregate metrics from")
    parser.add_argument("--metrics", nargs="+", default=METRICS, help="List of metric names to extract from wandb runs")
    parser.add_argument("--out_dir", type=str, default=os.path.join(os.path.dirname(__file__), "figures", "complexity"), help="Directory to save plots")
    parser.add_argument("--acc_vmin", type=float, default=50.0, help="Default heatmap minimum for accuracy plots.")
    parser.add_argument("--acc_vmax", type=float, default=100.0, help="Default heatmap maximum for accuracy plots.")
    parser.add_argument("--u2u_vmin", type=float, default=60.0, help="Heatmap minimum for the u->u accuracy plot.")
    parser.add_argument("--u2u_vmax", type=float, default=80.0, help="Heatmap maximum for the u->u accuracy plot.")
    parser.add_argument("--u2s_vmin", type=float, default=50.0, help="Heatmap minimum for the u->s accuracy plot.")
    parser.add_argument("--u2s_vmax", type=float, default=70.0, help="Heatmap maximum for the u->s accuracy plot.")
    parser.add_argument("--s2s_vmin", type=float, default=70.0, help="Heatmap minimum for the s->s accuracy plot.")
    parser.add_argument("--s2s_vmax", type=float, default=95.0, help="Heatmap maximum for the s->s accuracy plot.")
    parser.add_argument("--s2u_vmin", type=float, default=50.0, help="Heatmap minimum for the s->u accuracy plot.")
    parser.add_argument("--s2u_vmax", type=float, default=70.0, help="Heatmap maximum for the s->u accuracy plot.")
    args = parser.parse_args()
    metric_bounds = _resolve_metric_bounds(args)

    api = wandb.Api()
    rows = []

    for project in args.projects:
        runs = api.runs(f"{args.entity}/{project}")
        for run in runs:
            row = {
                "project": project,
                "run_id": run.id,
                "run_name": run.name,
                "state": run.state,
            }
            for metric in args.metrics:
                row[metric] = run.summary.get(metric, None)
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.dropna(subset=args.metrics, how="all")

    print(df.head())
    print(df.columns.tolist())

    for metric_name, (mean_col, std_col) in ACC_METRICS.items():
        vmin, vmax = metric_bounds[metric_name]
        _plot_metric(
            df=df,
            out_dir=args.out_dir,
            metric_name=metric_name,
            mean_col=mean_col,
            std_col=std_col,
            vmin=vmin,
            vmax=vmax,
            use_heatmap=True,
            use_model_colors=False,
        )

    _plot_delta_to_ideal(df=df, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
