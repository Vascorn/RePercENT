import argparse
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import wandb

ENTITY = "vasiliki-rizou-epfl"

PROJECTS = [
    "posthoc_synthetic-moe-2M",
    "posthoc_synthetic-moe-3M",
    "posthoc_synthetic-moe-4M",
    "posthoc_synthetic-moe-5M",
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

MODEL_MARKERS = {
    "RePercENT": "o",
    "gMLP": "s",
    "GRU": "^",
    "MLP": "D",
    "JointOpt": "P",
    "Other": "X",
}


def _extract_num_modalities(project_name: str) -> int | None:
    match = re.search(r"-(\d+)M$", project_name)
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


def _plot_metric(df: pd.DataFrame, out_dir: str, metric_name: str, mean_col: str, std_col: str) -> None:
    grouped = _aggregate_plot_df(df, mean_col, std_col)
    if grouped.empty:
        print(f"No rows available for plotting {metric_name}.")
        return

    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.6, 5.8))

    vmin = 50.0
    vmax = 100.0
    cmap = plt.get_cmap("viridis")
    label_offsets = {
        "RePercENT": (0, 11),
        "gMLP": (0, -13),
        "GRU": (12, 2),
        "MLP": (-20, 2),
        "JointOpt": (12, 11),
        "Other": (-18, -11),
    }
    scatter_ref = None

    for family_name, family_df in grouped.groupby("model_family"):
        marker = MODEL_MARKERS.get(family_name, MODEL_MARKERS["Other"])
        is_repercent = family_name == "RePercENT"
        scatter_ref = ax.scatter(
            family_df["num_modalities"],
            family_df["model_params"],
            c=family_df["metric_mean"],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            marker=marker,
            s=110 if is_repercent else 72,
            edgecolors="black" if is_repercent else "white",
            linewidths=1.1 if is_repercent else 0.5,
            alpha=1.0 if is_repercent else 0.9,
            zorder=4 if is_repercent else 3,
        )

        dx, dy = label_offsets.get(family_name, (6, 6))
        for _, row in family_df.iterrows():
            ax.annotate(
                f"{row['metric_mean']:.1f}%",
                (row["num_modalities"], row["model_params"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8 if is_repercent else 7,
                color="black" if is_repercent else "dimgray",
                alpha=1.0 if is_repercent else 0.82,
                weight="bold" if is_repercent else None,
            )

    ax.set_title(f"{metric_name.replace('_', ' ').upper()} vs Complexity")
    ax.set_xlabel("Number of Modalities")
    ax.set_ylabel("Number of Parameters")
    modality_ticks = sorted(grouped["num_modalities"].unique())
    ax.set_xticks(modality_ticks)
    ax.set_xlim(min(modality_ticks) - 0.35, max(modality_ticks) + 0.35)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cbar = fig.colorbar(scatter_ref, ax=ax)
    cbar.set_label("Accuracy")

    legend_handles = []
    legend_labels = []
    for family_name in ["RePercENT", "gMLP", "GRU", "MLP", "JointOpt", "Other"]:
        if family_name not in grouped["model_family"].unique():
            continue
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=MODEL_MARKERS.get(family_name, MODEL_MARKERS["Other"]),
                linestyle="None",
                markerfacecolor="lightgray",
                markeredgecolor="black" if family_name == "RePercENT" else "gray",
                markeredgewidth=1.1 if family_name == "RePercENT" else 0.6,
                markersize=7,
                color="none",
            )
        )
        legend_labels.append(family_name)
    ax.legend(legend_handles, legend_labels, title="Model", loc="upper left", frameon=True)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{metric_name}_modalities_params_heatmap.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate metrics from multiple wandb projects")
    parser.add_argument("--entity", type=str, default=ENTITY, help="Wandb entity name")
    parser.add_argument("--projects", nargs="+", default=PROJECTS, help="List of wandb project names to aggregate metrics from")
    parser.add_argument("--metrics", nargs="+", default=METRICS, help="List of metric names to extract from wandb runs")
    parser.add_argument("--out_dir", type=str, default=os.path.join(os.path.dirname(__file__), "figures", "complexity"), help="Directory to save plots")
    args = parser.parse_args()

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
        _plot_metric(df=df, out_dir=args.out_dir, metric_name=metric_name, mean_col=mean_col, std_col=std_col)


if __name__ == "__main__":
    main()
