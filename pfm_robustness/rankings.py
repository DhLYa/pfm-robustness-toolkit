"""Cross-dataset model rankings (dissertation Tables 5 and 11).

Within each dataset, models are ranked by score (competition ranking: ties share the lower
rank). Across datasets, scores, stds and ranks are averaged.
"""

import numpy as np
import pandas as pd

from .paths import stats_dir


def rank_desc(scores: pd.Series) -> pd.Series:
    """Higher is better; ties share a rank and the next rank is skipped (method='min')."""
    return scores.rank(ascending=False, method="min")


def average_across_datasets(per_dataset: pd.DataFrame, value_col: str, std_col: str, rank_col: str,
                            out_names=("avg_value", "avg_std", "avg_rank")) -> pd.DataFrame:
    """Per-model means of value, std and rank over datasets, sorted by average rank."""
    value_name, std_name, rank_name = out_names
    agg = (
        per_dataset.groupby("model", as_index=False)
        .agg(**{
            value_name: (value_col, "mean"),
            std_name: (std_col, "mean"),
            rank_name: (rank_col, lambda s: s.astype(float).mean()),
        })
        .sort_values(rank_name, ascending=True)
    )
    return agg


# ─── PPI ─────────────────────────────────────────────────────────────────────

def _mean_row(df: pd.DataFrame, label="Mean") -> pd.Series:
    """Row labelled `label` (as index value or in the first column) -> Series over models."""
    idx = df.index.astype(str).str.strip().str.lower()
    if (idx == label.lower()).any():
        return pd.to_numeric(df.loc[idx == label.lower()].iloc[0], errors="coerce")
    first = df.columns[0]
    mask = df[first].astype(str).str.strip().str.lower() == label.lower()
    if mask.any():
        return pd.to_numeric(df.loc[mask].iloc[0].drop(labels=[first], errors="ignore"), errors="coerce")
    raise KeyError("Could not find a row labelled 'Mean' in the sheet.")


def read_ppi_for_dataset(dataset: str) -> pd.DataFrame:
    """Columns: dataset, model, ppi_mean, ppi_std, rank."""
    xlsx = stats_dir(dataset, "constraint_sweep", "ppi_aggregate.xlsx")
    if not xlsx.is_file():
        raise FileNotFoundError(f"Missing: {xlsx}")
    s_mean = _mean_row(pd.read_excel(xlsx, sheet_name="ppi_mean", header=0))
    s_std = _mean_row(pd.read_excel(xlsx, sheet_name="ppi_std", header=0))

    models = sorted(set(s_mean.index.astype(str)).union(s_std.index.astype(str)))
    df = pd.DataFrame([{
        "dataset": dataset,
        "model": m,
        "ppi_mean": float(s_mean.get(m, np.nan)) if pd.notna(s_mean.get(m, np.nan)) else np.nan,
        "ppi_std": float(s_std.get(m, np.nan)) if pd.notna(s_std.get(m, np.nan)) else np.nan,
    } for m in models])
    df["rank"] = rank_desc(df["ppi_mean"]) if df["ppi_mean"].notna().any() else np.nan
    return df


def ppi_rankings(datasets) -> pd.DataFrame:
    """Writes Stats/PPI_aggregate_model_rankings.xlsx; returns the aggregate table."""
    per_ds = []
    for ds in datasets:
        try:
            per_ds.append(read_ppi_for_dataset(ds))
        except Exception as e:
            print(f"Skipped {ds}: {e}")
    if not per_ds:
        raise RuntimeError("No datasets processed; run the constraint sweeps + PPI first.")
    per_ds_df = pd.concat(per_ds, ignore_index=True)

    agg = average_across_datasets(per_ds_df, "ppi_mean", "ppi_std", "rank",
                                  out_names=("avg_PPI", "avg_std", "avg_rank"))
    agg = agg[["avg_rank", "model", "avg_PPI", "avg_std"]]

    out_xlsx = stats_dir("PPI_aggregate_model_rankings.xlsx")
    with pd.ExcelWriter(out_xlsx) as xw:
        agg.to_excel(xw, sheet_name="AGGREGATE_RANKINGS", index=False, float_format="%.6f")
        per_ds_df.sort_values(["dataset", "rank", "ppi_mean"], ascending=[True, True, False]) \
                 .to_excel(xw, sheet_name="PER_DATASET_RANKS", index=False, float_format="%.6f")
    print(f"Saved: {out_xlsx}")
    return agg
