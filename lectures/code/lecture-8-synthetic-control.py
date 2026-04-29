"""Self-contained synthetic-control example for Lecture 8.

The script simulates one treated unit and several donor units. It then estimates
convex synthetic-control weights by constrained least squares using projected
gradient descent. No external data are downloaded.

Run:
    python lectures/code/lecture-8-synthetic-control.py
    python lectures/code/lecture-8-synthetic-control.py --plot
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class SyntheticResult:
    weights: np.ndarray
    synthetic_path: np.ndarray
    gap: np.ndarray
    pre_rmspe: float
    post_rmspe: float

    @property
    def rmspe_ratio(self) -> float:
        return self.post_rmspe / max(self.pre_rmspe, 1e-12)


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Project a vector onto {w: w >= 0, sum(w) = 1}."""
    if v.ndim != 1:
        raise ValueError("project_to_simplex expects a one-dimensional vector")

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = u * np.arange(1, len(u) + 1) > (cssv - 1.0)
    if not np.any(rho_candidates):
        return np.ones_like(v) / len(v)

    rho = np.nonzero(rho_candidates)[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def fit_convex_weights(
    X_pre: np.ndarray,
    y_pre: np.ndarray,
    max_iter: int = 20_000,
    tol: float = 1e-12,
) -> np.ndarray:
    """Fit weights minimizing ||y_pre - X_pre @ w||^2 subject to simplex constraints."""
    n_periods, n_donors = X_pre.shape
    if y_pre.shape != (n_periods,):
        raise ValueError("y_pre must be a one-dimensional vector with X_pre rows")

    spectral_norm = np.linalg.norm(X_pre, ord=2)
    lipschitz = (spectral_norm**2) / n_periods
    step = 1.0 / max(lipschitz, 1e-12)

    weights = np.ones(n_donors) / n_donors
    for _ in range(max_iter):
        residual = X_pre @ weights - y_pre
        gradient = (X_pre.T @ residual) / n_periods
        new_weights = project_to_simplex(weights - step * gradient)
        if np.linalg.norm(new_weights - weights, ord=1) < tol:
            weights = new_weights
            break
        weights = new_weights

    return weights


def rmspe(gap: np.ndarray) -> float:
    return float(np.sqrt(np.mean(gap**2)))


def synthetic_control(
    outcomes: np.ndarray,
    treated_index: int,
    donor_indices: Iterable[int],
    treatment_start: int,
) -> SyntheticResult:
    """Estimate a synthetic control for one unit using selected donor indices."""
    donors = np.array(list(donor_indices), dtype=int)
    y = outcomes[:, treated_index]
    X = outcomes[:, donors]

    weights = fit_convex_weights(X[:treatment_start], y[:treatment_start])
    synthetic_path = X @ weights
    gap = y - synthetic_path

    return SyntheticResult(
        weights=weights,
        synthetic_path=synthetic_path,
        gap=gap,
        pre_rmspe=rmspe(gap[:treatment_start]),
        post_rmspe=rmspe(gap[treatment_start:]),
    )


def make_simulated_panel(
    seed: int = 123,
    n_donors: int = 10,
    n_periods: int = 35,
    treatment_start: int = 22,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a panel with one treated unit in column 0 and donors in columns 1..J."""
    rng = np.random.default_rng(seed)
    time = np.arange(n_periods)

    common_trend = 50.0 + 0.65 * time
    business_cycle = 3.0 * np.sin(time / 3.5)
    late_shock = np.maximum(time - 14, 0) * 0.18

    donors = []
    for _ in range(n_donors):
        level = rng.normal(0.0, 5.0)
        slope = rng.normal(0.0, 0.18)
        cycle_loading = rng.normal(1.0, 0.25)
        shock_loading = rng.normal(1.0, 0.30)
        noise = rng.normal(0.0, 0.65, size=n_periods)
        donor_path = (
            common_trend
            + level
            + slope * time
            + cycle_loading * business_cycle
            + shock_loading * late_shock
            + noise
        )
        donors.append(donor_path)

    donor_matrix = np.column_stack(donors)
    true_weights = np.zeros(n_donors)
    active = np.array([1, 3, 6, 8])
    true_weights[active] = np.array([0.30, 0.25, 0.25, 0.20])

    untreated_treated_path = donor_matrix @ true_weights
    untreated_treated_path += rng.normal(0.0, 0.45, size=n_periods)

    treatment_effect = np.zeros(n_periods)
    post_time = np.arange(n_periods - treatment_start)
    treatment_effect[treatment_start:] = -1.5 - 0.75 * post_time

    observed_treated_path = untreated_treated_path + treatment_effect
    outcomes = np.column_stack([observed_treated_path, donor_matrix])
    return outcomes, untreated_treated_path, treatment_effect, true_weights


def placebo_results(outcomes: np.ndarray, treatment_start: int) -> list[SyntheticResult]:
    """Run placebo synthetic controls for donor units, excluding the real treated unit."""
    n_units = outcomes.shape[1]
    results: list[SyntheticResult] = []
    for placebo_index in range(1, n_units):
        donor_indices = [j for j in range(1, n_units) if j != placebo_index]
        results.append(
            synthetic_control(
                outcomes=outcomes,
                treated_index=placebo_index,
                donor_indices=donor_indices,
                treatment_start=treatment_start,
            )
        )
    return results


def print_summary(
    treated_result: SyntheticResult,
    placebo: list[SyntheticResult],
    treatment_start: int,
    true_weights: np.ndarray,
) -> None:
    final_gap = treated_result.gap[-1]
    placebo_final_gaps = np.array([result.gap[-1] for result in placebo])
    placebo_p_value = (1.0 + np.sum(placebo_final_gaps <= final_gap)) / (
        1.0 + len(placebo_final_gaps)
    )

    active_weights = [
        (donor_id + 1, weight)
        for donor_id, weight in enumerate(treated_result.weights)
        if weight > 0.01
    ]
    true_active_weights = [
        (donor_id + 1, weight)
        for donor_id, weight in enumerate(true_weights)
        if weight > 0.0
    ]

    print("Lecture 8 synthetic-control simulation")
    print("--------------------------------------")
    print(f"Treatment starts at period index: {treatment_start}")
    print(f"Pre-treatment RMSPE: {treated_result.pre_rmspe:0.3f}")
    print(f"Post-treatment RMSPE: {treated_result.post_rmspe:0.3f}")
    print(f"RMSPE ratio: {treated_result.rmspe_ratio:0.2f}")
    print(f"Final-period estimated effect: {final_gap:0.3f}")
    print(f"One-sided placebo p-value for a negative effect: {placebo_p_value:0.3f}")
    print()
    print("Estimated donor weights above 1 percent:")
    for donor_id, weight in active_weights:
        print(f"  donor {donor_id:02d}: {weight:0.3f}")
    print()
    print("True data-generating donor weights:")
    for donor_id, weight in true_active_weights:
        print(f"  donor {donor_id:02d}: {weight:0.3f}")


def plot_results(
    outcomes: np.ndarray,
    treated_result: SyntheticResult,
    placebo: list[SyntheticResult],
    treatment_start: int,
    true_effect: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    time = np.arange(outcomes.shape[0])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    axes[0].plot(time, outcomes[:, 0], label="treated", linewidth=2.5)
    axes[0].plot(time, treated_result.synthetic_path, label="synthetic", linewidth=2.5)
    axes[0].axvline(treatment_start, color="black", linestyle=":", linewidth=1.5)
    axes[0].set_title("Treated vs. synthetic path")
    axes[0].set_xlabel("period")
    axes[0].set_ylabel("outcome")
    axes[0].legend()

    axes[1].plot(time, treated_result.gap, label="estimated gap", linewidth=2.5)
    axes[1].plot(time, true_effect, label="true effect", linestyle="--", linewidth=2)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].axvline(treatment_start, color="black", linestyle=":", linewidth=1.5)
    axes[1].set_title("Effect path")
    axes[1].set_xlabel("period")
    axes[1].legend()

    for result in placebo:
        axes[2].plot(time, result.gap, color="gray", alpha=0.35, linewidth=1)
    axes[2].plot(time, treated_result.gap, color="tab:blue", linewidth=2.5, label="treated")
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].axvline(treatment_start, color="black", linestyle=":", linewidth=1.5)
    axes[2].set_title("Placebo gaps")
    axes[2].set_xlabel("period")
    axes[2].legend()

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="show diagnostic plots")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument("--donors", type=int, default=10, help="number of donor units")
    args = parser.parse_args()

    treatment_start = 22
    outcomes, _, true_effect, true_weights = make_simulated_panel(
        seed=args.seed,
        n_donors=args.donors,
        treatment_start=treatment_start,
    )

    treated_result = synthetic_control(
        outcomes=outcomes,
        treated_index=0,
        donor_indices=range(1, outcomes.shape[1]),
        treatment_start=treatment_start,
    )
    placebo = placebo_results(outcomes, treatment_start=treatment_start)

    print_summary(
        treated_result=treated_result,
        placebo=placebo,
        treatment_start=treatment_start,
        true_weights=true_weights,
    )

    if args.plot:
        plot_results(
            outcomes=outcomes,
            treated_result=treated_result,
            placebo=placebo,
            treatment_start=treatment_start,
            true_effect=true_effect,
        )


if __name__ == "__main__":
    main()
