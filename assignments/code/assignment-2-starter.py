"""Create starter datasets for Assignment 2.

Run from the course root:

    python assignments/code/assignment-2-starter.py

The script writes three CSV files to assignments/data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def make_iv_sample(*, weak: bool, seed: int, n: int = 2_000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    person_id = np.arange(1, n + 1)
    offer = rng.binomial(1, 0.50, n)
    ability = rng.normal(0.0, 1.0, n)
    age = rng.integers(22, 56, n)
    baseline_earnings = 28_000 + 3_500 * ability + rng.normal(0.0, 4_000, n)

    first_stage_strength = 0.85 if not weak else 0.12
    takeup_index = (
        -0.25
        + first_stage_strength * offer
        + 0.70 * ability
        + 0.000015 * (baseline_earnings - 28_000)
        + rng.normal(0.0, 1.0, n)
    )
    training = (takeup_index > 0.0).astype(int)

    earnings = (
        32_000
        + 4_500 * training
        + 5_500 * ability
        + 0.35 * baseline_earnings
        + 80 * (age - 35)
        + rng.normal(0.0, 5_000, n)
    )

    return pd.DataFrame(
        {
            "person_id": person_id,
            "earnings": earnings,
            "training": training,
            "offer": offer,
            "age": age,
            "baseline_earnings": baseline_earnings,
        }
    )


def make_matching_sample(seed: int, n: int = 1_800) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    person_id = np.arange(1, n + 1)
    female = rng.binomial(1, 0.48, n)
    age = rng.integers(22, 58, n)
    education = np.clip(np.round(rng.normal(13.0, 2.2, n)), 8, 20)
    pre_wage = 16.0 + 1.2 * (education - 12) + 0.05 * (age - 35) - 0.8 * female
    pre_wage = pre_wage + rng.normal(0.0, 2.5, n)

    program_probability = sigmoid(
        -0.40
        + 0.28 * (education - 12)
        - 0.05 * (age - 35)
        + 0.16 * pre_wage
        + 0.20 * female
    )
    program = rng.binomial(1, program_probability)

    treatment_effect = 2.2 + 0.30 * (education - 12) + 0.45 * female
    untreated_wage = (
        18.0
        + 1.1 * (education - 12)
        + 0.08 * (age - 35)
        + 0.65 * pre_wage
        - 0.7 * female
        + rng.normal(0.0, 2.8, n)
    )
    wage = untreated_wage + program * treatment_effect

    return pd.DataFrame(
        {
            "person_id": person_id,
            "wage": wage,
            "program": program,
            "age": age,
            "education": education.astype(int),
            "pre_wage": pre_wage,
            "female": female,
        }
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "assignment2_iv.csv": make_iv_sample(weak=False, seed=20260503),
        "assignment2_iv_weak.csv": make_iv_sample(weak=True, seed=20260504),
        "assignment2_matching.csv": make_matching_sample(seed=20260505),
    }

    for filename, df in datasets.items():
        path = DATA_DIR / filename
        df.to_csv(path, index=False)
        print(f"Wrote {path} with shape {df.shape}")


if __name__ == "__main__":
    main()
