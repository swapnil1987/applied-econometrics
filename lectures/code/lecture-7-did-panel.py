"""
Lecture 7 simulation: difference-in-differences and panel fixed effects.

This script creates two panel datasets:

1. Parallel trends holds.
2. Parallel trends fails because treated units already have a steeper
   untreated trend.

It then computes:
- the 2-by-2 DiD estimate;
- the two-way fixed effects estimate;
- simple event-study coefficients.

The script uses simulated data only. It writes no files unless --save-figures
is supplied.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ScenarioResult:
    name: str
    data: pd.DataFrame
    did: float
    twfe: float
    true_effect: float
    event_study: pd.DataFrame


def simulate_panel(
    *,
    parallel_trends: bool,
    n_units: int = 600,
    n_periods: int = 8,
    treatment_start: int = 5,
    tau: float = 2.0,
    seed: int = 123,
) -> pd.DataFrame:
    """Simulate a balanced panel with one treated cohort and never-treated controls."""
    rng = np.random.default_rng(seed)

    units = np.arange(n_units)
    periods = np.arange(n_periods)
    treated_unit = units >= n_units // 2

    unit_fe = rng.normal(loc=0.0, scale=1.0, size=n_units)
    # A common time path shared by treated and controls.
    time_fe = 0.35 * periods + 0.12 * np.sin(periods)

    rows = []
    for i in units:
        for t in periods:
            treated_group = int(treated_unit[i])
            post = int(t >= treatment_start)
            treatment = treated_group * post

            untreated_trend_violation = 0.0
            if not parallel_trends:
                # This is the violation: treated units would have grown faster
                # even if the policy had never happened.
                untreated_trend_violation = 0.28 * treated_group * t

            y0 = (
                10.0
                + 1.4 * treated_group
                + unit_fe[i]
                + time_fe[t]
                + untreated_trend_violation
                + rng.normal(loc=0.0, scale=0.7)
            )
            y = y0 + tau * treatment

            rows.append(
                {
                    "unit": i,
                    "time": t,
                    "treated_group": treated_group,
                    "post": post,
                    "treatment": treatment,
                    "event_time": t - treatment_start if treated_group else np.nan,
                    "y0": y0,
                    "y": y,
                    "true_tau": tau if treatment else 0.0,
                }
            )

    return pd.DataFrame(rows)


def did_2x2(df: pd.DataFrame) -> float:
    """Compute the 2-by-2 DiD estimate from group-period means."""
    means = (
        df.groupby(["treated_group", "post"], as_index=True)["y"]
        .mean()
        .unstack("post")
    )
    treated_change = means.loc[1, 1] - means.loc[1, 0]
    control_change = means.loc[0, 1] - means.loc[0, 0]
    return float(treated_change - control_change)


def ols(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return least-squares coefficients."""
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return beta


def design_matrix_twfe(df: pd.DataFrame, treatment_col: str = "treatment") -> tuple[np.ndarray, list[str]]:
    """Build an OLS design matrix with treatment, unit FE, and time FE."""
    unit_dummies = pd.get_dummies(df["unit"], prefix="unit", drop_first=True, dtype=float)
    time_dummies = pd.get_dummies(df["time"], prefix="time", drop_first=True, dtype=float)

    x = pd.concat(
        [
            pd.Series(1.0, index=df.index, name="const"),
            df[[treatment_col]].astype(float),
            unit_dummies,
            time_dummies,
        ],
        axis=1,
    )
    return x.to_numpy(), list(x.columns)


def twfe_estimate(df: pd.DataFrame) -> float:
    """Estimate y on treatment plus unit and time fixed effects."""
    y = df["y"].to_numpy()
    x, names = design_matrix_twfe(df, treatment_col="treatment")
    beta = ols(y, x)
    return float(beta[names.index("treatment")])


def event_study(df: pd.DataFrame, treatment_start: int = 5, omit_event_time: int = -1) -> pd.DataFrame:
    """Estimate a simple event study with unit and time fixed effects.

    The coefficients are interactions between treated-group status and event
    time. The period immediately before treatment, k=-1, is omitted.
    """
    work = df.copy()
    treated_times = sorted(
        int(k)
        for k in work.loc[work["treated_group"].eq(1), "event_time"].dropna().unique()
        if int(k) != omit_event_time
    )

    event_cols = []
    for k in treated_times:
        col = f"event_{k:+d}"
        work[col] = ((work["treated_group"] == 1) & (work["time"] - treatment_start == k)).astype(float)
        event_cols.append(col)

    unit_dummies = pd.get_dummies(work["unit"], prefix="unit", drop_first=True, dtype=float)
    time_dummies = pd.get_dummies(work["time"], prefix="time", drop_first=True, dtype=float)
    x = pd.concat(
        [
            pd.Series(1.0, index=work.index, name="const"),
            work[event_cols],
            unit_dummies,
            time_dummies,
        ],
        axis=1,
    )

    beta = ols(work["y"].to_numpy(), x.to_numpy())
    names = list(x.columns)
    rows = []
    for k, col in zip(treated_times, event_cols):
        rows.append({"event_time": k, "coefficient": float(beta[names.index(col)])})
    return pd.DataFrame(rows)


def summarize_scenario(name: str, df: pd.DataFrame) -> ScenarioResult:
    did = did_2x2(df)
    twfe = twfe_estimate(df)
    true_effect = df.loc[df["treatment"].eq(1), "true_tau"].mean()
    es = event_study(df)
    return ScenarioResult(
        name=name,
        data=df,
        did=did,
        twfe=twfe,
        true_effect=float(true_effect),
        event_study=es,
    )


def print_summary(result: ScenarioResult) -> None:
    print("=" * 78)
    print(result.name)
    print("-" * 78)
    print(f"True post-treatment effect among treated: {result.true_effect: .3f}")
    print(f"2-by-2 DiD estimate:                  {result.did: .3f}")
    print(f"TWFE estimate:                        {result.twfe: .3f}")
    print()

    pre = result.event_study[result.event_study["event_time"] < 0]
    post = result.event_study[result.event_study["event_time"] >= 0]
    print("Event-study coefficients relative to k=-1")
    print("Pre-treatment leads:")
    print(pre.to_string(index=False, float_format=lambda x: f"{x: .3f}"))
    print("Post-treatment coefficients:")
    print(post.to_string(index=False, float_format=lambda x: f"{x: .3f}"))
    print()


def make_plots(results: list[ScenarioResult], *, save_figures: bool, show: bool) -> None:
    """Plot group means and event-study coefficients if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plots.")
        return

    fig, axes = plt.subplots(2, len(results), figsize=(12, 7), sharex=False)
    if len(results) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, result in enumerate(results):
        means = (
            result.data.groupby(["time", "treated_group"], as_index=False)["y"]
            .mean()
            .replace({"treated_group": {0: "Control", 1: "Treated"}})
        )
        for label, sub in means.groupby("treated_group"):
            axes[0, col].plot(sub["time"], sub["y"], marker="o", label=label)
        axes[0, col].axvline(4.5, color="black", linestyle="--", linewidth=1)
        axes[0, col].set_title(result.name)
        axes[0, col].set_xlabel("Time")
        axes[0, col].set_ylabel("Mean outcome")
        axes[0, col].legend()

        es = result.event_study
        axes[1, col].axhline(0, color="black", linewidth=1)
        axes[1, col].axvline(-0.5, color="black", linestyle="--", linewidth=1)
        axes[1, col].scatter(es["event_time"], es["coefficient"])
        axes[1, col].plot(es["event_time"], es["coefficient"], linewidth=1)
        axes[1, col].set_xlabel("Event time")
        axes[1, col].set_ylabel("Coefficient")

    fig.tight_layout()

    if save_figures:
        out = Path(__file__).with_name("lecture-7-did-panel-simulation.png")
        fig.savefig(out, dpi=200)
        print(f"Saved figure to {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="display matplotlib figures")
    parser.add_argument("--save-figures", action="store_true", help="save a PNG next to this script")
    args = parser.parse_args()

    parallel = simulate_panel(parallel_trends=True, seed=123)
    violation = simulate_panel(parallel_trends=False, seed=123)

    results = [
        summarize_scenario("Scenario 1: parallel trends holds", parallel),
        summarize_scenario("Scenario 2: parallel trends fails", violation),
    ]

    for result in results:
        print_summary(result)

    make_plots(results, save_figures=args.save_figures, show=args.show)


if __name__ == "__main__":
    main()
