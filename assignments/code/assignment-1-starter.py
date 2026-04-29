"""Create starter datasets for Assignment 1.

Run from the course root:

    python assignments/code/assignment-1-starter.py

The script writes two CSV files to assignments/data. Students should use the
CSV files for analysis, not the data-generating code as an answer key.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def make_sample(*, randomized: bool, seed: int, n: int = 1_200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    student_id = np.arange(1, n + 1)
    female = rng.binomial(1, 0.55, n)
    baseline_score = rng.normal(0.0, 1.0, n)
    parent_edu = np.clip(np.round(rng.normal(13.5, 2.4, n)), 8, 20)
    age = rng.integers(18, 31, n)

    # Motivation is intentionally unobserved in the CSV. In the observational
    # sample it affects both treatment take-up and outcomes.
    motivation = rng.normal(0.0, 1.0, n)

    if randomized:
        treatment_probability = np.full(n, 0.50)
    else:
        treatment_probability = sigmoid(
            -0.25
            + 0.75 * baseline_score
            + 0.10 * (parent_edu - 13.5)
            + 0.20 * female
            - 0.03 * (age - 23)
            + 0.65 * motivation
        )

    scholarship = rng.binomial(1, treatment_probability)

    untreated_score = (
        65.0
        + 5.0 * baseline_score
        + 0.9 * (parent_edu - 13.5)
        - 0.20 * (age - 23)
        + 2.4 * motivation
        + rng.normal(0.0, 5.0, n)
    )
    individual_effect = 4.0 + 0.8 * baseline_score + 0.4 * female
    final_score = untreated_score + scholarship * individual_effect

    study_hours_after = (
        7.0
        + 1.6 * scholarship
        + 0.7 * baseline_score
        + 0.5 * motivation
        + rng.normal(0.0, 1.8, n)
    )

    return pd.DataFrame(
        {
            "student_id": student_id,
            "final_score": final_score,
            "scholarship": scholarship,
            "female": female,
            "baseline_score": baseline_score,
            "parent_edu": parent_edu.astype(int),
            "age": age,
            "study_hours_after": study_hours_after,
        }
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "assignment1_observational.csv": make_sample(randomized=False, seed=20260501),
        "assignment1_randomized.csv": make_sample(randomized=True, seed=20260502),
    }

    for filename, df in datasets.items():
        path = DATA_DIR / filename
        df.to_csv(path, index=False)
        print(f"Wrote {path} with shape {df.shape}")


if __name__ == "__main__":
    main()
