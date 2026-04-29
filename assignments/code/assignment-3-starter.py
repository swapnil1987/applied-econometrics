"""Create starter panel datasets for Assignment 3.

Run from the course root:

    python assignments/code/assignment-3-starter.py

The script writes two CSV files to assignments/data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def make_panel(*, violate_parallel_trends: bool, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    n_units = 240
    years = np.arange(2016, 2024)
    first_treated_year = 2020
    units = np.arange(1, n_units + 1)
    treated_group_by_unit = (units > n_units // 2).astype(int)
    unit_fe = rng.normal(0.0, 2.0, n_units)
    calendar_trend = {year: 0.55 * (year - 2016) + 0.20 * np.sin(year) for year in years}

    rows: list[dict[str, float | int]] = []
    for unit, treated_group, alpha_i in zip(units, treated_group_by_unit, unit_fe):
        for year in years:
            post = int(year >= first_treated_year)
            treated_post = treated_group * post
            event_time = year - first_treated_year if treated_group else -999

            untreated_trend_gap = 0.0
            if violate_parallel_trends:
                untreated_trend_gap = 0.35 * treated_group * (year - 2016)

            y0 = (
                20.0
                + 2.5 * treated_group
                + alpha_i
                + calendar_trend[year]
                + untreated_trend_gap
                + rng.normal(0.0, 1.2)
            )
            outcome = y0 + 3.0 * treated_post

            rows.append(
                {
                    "unit": unit,
                    "year": year,
                    "treated_group": treated_group,
                    "post": post,
                    "treated_post": treated_post,
                    "event_time": event_time,
                    "outcome": outcome,
                }
            )

    return pd.DataFrame(rows)


def print_group_means(df: pd.DataFrame, label: str) -> None:
    means = (
        df.groupby(["treated_group", "post"], as_index=False)["outcome"]
        .mean()
        .sort_values(["treated_group", "post"])
    )
    print()
    print(label)
    print(means.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "assignment3_panel_clean.csv": make_panel(
            violate_parallel_trends=False,
            seed=20260506,
        ),
        "assignment3_panel_violation.csv": make_panel(
            violate_parallel_trends=True,
            seed=20260507,
        ),
    }

    for filename, df in datasets.items():
        path = DATA_DIR / filename
        df.to_csv(path, index=False)
        print(f"Wrote {path} with shape {df.shape}")
        print_group_means(df, filename)


if __name__ == "__main__":
    main()
