import argparse
import math
import os
import re
import time

import pandas as pd
import wandb


ENTITY = "vasiliki-rizou-epfl"
PROJECT = "repercent_ablation_synthetic"

ACC_METRICS = {
    "u2u_acc": ("u2u_acc_mean", "u2u_acc_std"),
    "u2s_acc": ("u2s_acc_mean", "u2s_acc_std"),
    "s2s_acc": ("s2s_acc_mean", "s2s_acc_std"),
    "s2u_acc": ("s2u_acc_mean", "s2u_acc_std"),
}

STRUCTURAL_CONFIGS = [
    ("w_SE", "w_GSA"),
    ("w_SE", "wo_GSA"),
    ("wo_SE", "w_GSA"),
    ("wo_SE", "wo_GSA"),
]


def _normalize_modalities(modalities: list[str]) -> list[str]:
    normalized = []
    for modality in modalities:
        for token in str(modality).split(","):
            token = token.strip()
            if not token:
                continue
            normalized.append(str(int(token)))
    return normalized


def _aggregate_run_name(num_modalities: str, se: str, gsa: str) -> str:
    return f"aggregate_repercent_splits_3_{num_modalities}M_{se}_{gsa}"


def _table_key(num_modalities: str, se: str, gsa: str) -> str:
    return f"repercent_splits_3_{num_modalities}M_{se}_{gsa}"


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


def _download_wandb_table(run, table_key: str, max_retries: int = 4) -> pd.DataFrame:
    artifacts = list(run.logged_artifacts())
    last_error = None
    for attempt in range(max_retries):
        for artifact in artifacts:
            if not _artifact_matches_table_key(artifact, table_key):
                continue
            for candidate_key in (table_key, table_key.replace(".", r"\.")):
                try:
                    return artifact.get(candidate_key).get_dataframe()
                except KeyError:
                    pass
                except Exception as exc:
                    last_error = exc
                    wait_seconds = 2 ** attempt
                    print(
                        f"W&B table download failed for {table_key!r} "
                        f"(attempt {attempt + 1}/{max_retries}): {exc}. "
                        f"Retrying in {wait_seconds}s."
                    )
                    time.sleep(wait_seconds)
                    break

    available = [str(getattr(artifact, "name", "")) for artifact in artifacts]
    if last_error is not None:
        raise RuntimeError(f"Could not download table {table_key!r} after {max_retries} attempts.") from last_error
    raise KeyError(
        f"Run {run.path} does not have a logged table for {table_key!r}. "
        f"Available artifacts: {available}"
    )


def _lookup_metric_row(table_df: pd.DataFrame, metric_key: str) -> pd.Series | None:
    if "metric" not in table_df.columns:
        raise ValueError(f"Expected a 'metric' column, got columns: {table_df.columns.tolist()}")

    metric_names = table_df["metric"].astype(str)
    exact = table_df[metric_names == metric_key]
    if len(exact) == 1:
        return exact.iloc[0]

    suffix = table_df[metric_names.str.endswith(f"/{metric_key}")]
    if len(suffix) == 1:
        return suffix.iloc[0]
    return None


def _calculate_metric_mean_std(table_df: pd.DataFrame, metric_key: str) -> tuple[float, float]:
    row = _lookup_metric_row(table_df, metric_key)
    if row is None:
        return math.nan, math.nan

    split_columns = sorted(
        [column for column in table_df.columns if re.fullmatch(r"split_\d+", str(column))],
        key=lambda column: int(str(column).split("_")[1]),
    )
    if split_columns:
        split_values = pd.to_numeric(row[split_columns], errors="coerce").dropna()
        if not split_values.empty:
            return float(split_values.mean()), float(split_values.std(ddof=0))

    if {"mean", "std"}.issubset(table_df.columns):
        return float(row["mean"]), float(row["std"])

    raise ValueError(
        f"Expected split columns or ['mean', 'std'] for metric {metric_key!r}, "
        f"got columns: {table_df.columns.tolist()}"
    )


def _format_mean_std(mean: float, std: float) -> str:
    if not math.isfinite(mean):
        return ""
    if not math.isfinite(std):
        return f"{mean:.2f}"
    return f"{mean:.2f} ± {std:.2f}"


def _run_created_at(run) -> pd.Timestamp:
    created_at = getattr(run, "created_at", None)
    if created_at is None:
        created_at = getattr(run, "createdAt", None)
    return pd.to_datetime(created_at, errors="coerce")


def _latest_run(runs, run_name: str):
    sorted_runs = sorted(
        list(runs),
        key=lambda run: (_run_created_at(run), str(getattr(run, "id", ""))),
        reverse=True,
    )
    latest = sorted_runs[0]
    if len(sorted_runs) > 1:
        print(
            f"Found {len(sorted_runs)} runs named {run_name}; "
            f"using latest created run {latest.id} from {getattr(latest, 'created_at', None)}."
        )
    return latest


def _calculate_config_metrics(table_df: pd.DataFrame, se: str, gsa: str) -> dict[str, str]:
    values = {}
    raw = {}
    for metric_name, (mean_key, _) in ACC_METRICS.items():
        mean, std = _calculate_metric_mean_std(table_df, mean_key)
        raw[f"{metric_name}_mean"] = mean
        raw[f"{metric_name}_std"] = std
        values[metric_name] = _format_mean_std(mean, std)

    delta_u_mean = raw["u2u_acc_mean"] - raw["u2s_acc_mean"]
    delta_u_std = math.sqrt(raw["u2u_acc_std"] ** 2 + raw["u2s_acc_std"] ** 2)
    delta_s_mean = raw["s2s_acc_mean"] - raw["s2u_acc_mean"]
    delta_s_std = math.sqrt(raw["s2s_acc_std"] ** 2 + raw["s2u_acc_std"] ** 2)

    values["delta_u"] = _format_mean_std(delta_u_mean, delta_u_std)
    values["delta_s"] = _format_mean_std(delta_s_mean, delta_s_std)
    values["configuration"] = f"{se}_{gsa}"
    return values


def _fetch_structural_rows(api: wandb.Api, entity: str, project: str, num_modalities: str) -> pd.DataFrame:
    rows = []
    for se, gsa in STRUCTURAL_CONFIGS:
        run_name = _aggregate_run_name(num_modalities, se, gsa)
        table_key = _table_key(num_modalities, se, gsa)
        runs = api.runs(f"{entity}/{project}", filters={"display_name": run_name})
        if len(runs) == 0:
            print(f"Missing aggregate run: {run_name}")
            rows.append({"configuration": f"{se}_{gsa}"})
            continue

        run = _latest_run(runs, run_name)
        table_df = _download_wandb_table(run, table_key)
        rows.append(_calculate_config_metrics(table_df, se, gsa))

    columns = ["u2u_acc", "u2s_acc", "s2s_acc", "s2u_acc", "delta_u", "delta_s"]
    df = pd.DataFrame(rows).set_index("configuration")
    return df.reindex([f"{se}_{gsa}" for se, gsa in STRUCTURAL_CONFIGS])[columns]


def main():
    parser = argparse.ArgumentParser(description="Export structural component ablation metrics from W&B tables.")
    parser.add_argument("--entity", type=str, default=ENTITY, help="W&B entity name")
    parser.add_argument("--project", type=str, default=PROJECT, help="W&B project name")
    parser.add_argument("--out_dir", type=str, default=os.path.join(os.path.dirname(__file__), "figures", "structural_comp_ablations"), help="Directory to save CSV files")
    parser.add_argument("--modalities", nargs="+", default=["2", "3", "4", "5"], help="Modality counts to export")

    args = parser.parse_args()
    modalities = _normalize_modalities(args.modalities)

    api = wandb.Api(timeout=60)
    os.makedirs(args.out_dir, exist_ok=True)

    for num_modalities in modalities:
        df = _fetch_structural_rows(api, args.entity, args.project, num_modalities)
        out_path = os.path.join(args.out_dir, f"structural_comp_ablation_{num_modalities}M.csv")
        df.to_csv(out_path, index_label="configuration")
        print(f"\nM={num_modalities}")
        print(df.to_string())
        print(f"Saved CSV to {out_path}")


if __name__ == "__main__":
    main()
