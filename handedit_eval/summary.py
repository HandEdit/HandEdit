from __future__ import annotations

import json
from typing import Iterable, List, Sequence

import pandas as pd


def jsonl_to_dataframe(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def summarize_dataframe(df: pd.DataFrame, group_by: Sequence[str], metric_columns: Sequence[str]) -> pd.DataFrame:
    rows = []
    for keys, sub_df in df.groupby(list(group_by), dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = {column: value for column, value in zip(group_by, keys)}
        row["n"] = int(len(sub_df))
        for metric in metric_columns:
            if metric not in sub_df.columns:
                continue
            values = _numeric(sub_df[metric]).dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else float("nan")
            row[f"{metric}_std"] = float(values.std()) if len(values) else float("nan")
            row[f"{metric}_median"] = float(values.median()) if len(values) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def save_leaderboard(summary_df: pd.DataFrame, csv_path: str, md_path: str, sort_key: str | None = None, ascending: bool = False) -> None:
    df = summary_df.copy()
    if sort_key and sort_key in df.columns:
        df = df.sort_values(sort_key, ascending=ascending)
    df.to_csv(csv_path, index=False)

    selected_columns = [
        column
        for column in df.columns
        if column in {"exp_name", "dataset", "task", "n", "Full-FID", "ROI-FID"} or column.endswith("_mean")
    ]
    if "exp_name" in df.columns and "exp_name" not in selected_columns:
        selected_columns = ["exp_name"] + selected_columns
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(df[selected_columns].to_markdown(index=False) + "\n")
