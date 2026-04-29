"""
Lecture 9 simulation: regression discontinuity design.

This script uses simulated data only. It demonstrates:
- sharp RD with a known treatment effect at the cutoff;
- local linear RD estimates with triangular kernel weights;
- fuzzy RD as a local Wald estimator;
- covariate balance and density/manipulation diagnostics.

The script writes no files unless --save-plots is supplied.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LocalLinearResult:
    bandwidth: float
    n_eff: int
    n_left: int
    n_right: int
    params: np.ndarray
    covariance: np.ndarray
    jump: float
    se: float
    ci_low: float
    ci_high: float


@dataclass
class FuzzyResult:
    bandwidth: float
    reduced_form: LocalLinearResult
    first_stage: LocalLinearResult
    wald: float
    bootstrap_se: float
    ci_low: float
    ci_high: float


def triangular_kernel(x: np.ndarray, bandwidth: float) -> np.ndarray:
    """Triangular kernel centered at zero."""
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    return np.maximum(0.0, 1.0 - np.abs(x) / bandwidth)


def rd_design_matrix(x: np.ndarray) -> np.ndarray:
    """Design matrix for a local linear RD with different slopes on each side."""
    z = (x >= 0).astype(float)
    return np.column_stack([np.ones_like(x), x, z, z * x])


def weighted_ols(y: np.ndarray, xmat: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted least squares with Eicker-Huber-White robust covariance."""
    mask = np.isfinite(y) & np.all(np.isfinite(xmat), axis=1) & np.isfinite(weights) & (weights > 0)
    y_use = y[mask]
    x_use = xmat[mask]
    w_use = weights[mask]

    if y_use.size <= x_use.shape[1]:
        raise ValueError("not enough observations with positive weight")

    xtwx = x_use.T @ (w_use[:, None] * x_use)
    bread = np.linalg.pinv(xtwx)
    beta = bread @ (x_use.T @ (w_use * y_use))
    resid = y_use - x_use @ beta

    meat = x_use.T @ ((w_use**2 * resid**2)[:, None] * x_use)
    cov = bread @ meat @ bread

    n_obs, n_par = x_use.shape
    if n_obs > n_par:
        cov *= n_obs / (n_obs - n_par)

    return beta, cov


def fit_local_linear(
    y: np.ndarray,
    running: np.ndarray,
    *,
    cutoff: float = 0.0,
    bandwidth: float = 0.75,
) -> LocalLinearResult:
    """Estimate a sharp RD jump using local linear WLS."""
    x = running - cutoff
    weights = triangular_kernel(x, bandwidth)
    xmat = rd_design_matrix(x)
    beta, cov = weighted_ols(y, xmat, weights)

    positive_weight = weights > 0
    n_left = int(np.sum(positive_weight & (x < 0)))
    n_right = int(np.sum(positive_weight & (x >= 0)))
    jump = float(beta[2])
    se = float(np.sqrt(max(cov[2, 2], 0.0)))

    return LocalLinearResult(
        bandwidth=bandwidth,
        n_eff=int(np.sum(positive_weight)),
        n_left=n_left,
        n_right=n_right,
        params=beta,
        covariance=cov,
        jump=jump,
        se=se,
        ci_low=jump - 1.96 * se,
        ci_high=jump + 1.96 * se,
    )


def predict_local_linear(result: LocalLinearResult, x: np.ndarray) -> np.ndarray:
    """Predict from a fitted local linear RD model at centered running variable x."""
    return rd_design_matrix(x) @ result.params


def simulate_sharp_rd(n: int = 4_000, seed: int = 123) -> dict[str, np.ndarray | float]:
    """Simulate a sharp RD with smooth potential outcomes."""
    rng = np.random.default_rng(seed)
    running = rng.uniform(-2.5, 2.5, size=n)
    treatment = (running >= 0).astype(float)

    y0 = (
        10.0
        + 2.0 * running
        + 1.5 * running**2
        - 0.45 * running**3
        + 0.8 * np.sin(2.0 * running)
    )
    tau_at_cutoff = 4.0
    heterogeneous_tau = tau_at_cutoff + 0.45 * running
    y = y0 + heterogeneous_tau * treatment + rng.normal(0.0, 1.5, size=n)

    # A predetermined covariate that is smooth at the cutoff.
    covariate = 5.0 + 0.6 * running + 0.2 * running**2 + rng.normal(0.0, 1.0, size=n)

    return {
        "running": running,
        "treatment": treatment,
        "y": y,
        "y0": y0,
        "covariate": covariate,
        "tau_at_cutoff": tau_at_cutoff,
    }


def simulate_fuzzy_rd(n: int = 4_000, seed: int = 321) -> dict[str, np.ndarray | float]:
    """Simulate a fuzzy RD with monotone treatment take-up."""
    rng = np.random.default_rng(seed)
    running = rng.uniform(-2.2, 2.2, size=n)
    cutoff_instrument = (running >= 0).astype(float)
    latent_resistance = rng.uniform(0.0, 1.0, size=n)

    p0 = np.clip(0.22 + 0.06 * running, 0.05, 0.55)
    p1 = np.clip(p0 + 0.45, 0.05, 0.95)

    treatment_below = (latent_resistance < p0).astype(float)
    treatment_above = (latent_resistance < p1).astype(float)
    treatment = np.where(cutoff_instrument == 1, treatment_above, treatment_below)
    complier = treatment_above > treatment_below

    tau = 6.0 + 1.5 * (latent_resistance < 0.35)
    y0 = 20.0 + 1.4 * running + 0.9 * running**2 - 0.25 * running**3
    y = y0 + tau * treatment + rng.normal(0.0, 2.0, size=n)

    local_compliers = complier & (np.abs(running) < 0.10)
    true_local_late = float(np.mean(tau[local_compliers]))

    return {
        "running": running,
        "instrument": cutoff_instrument,
        "treatment": treatment,
        "y": y,
        "tau": tau,
        "complier": complier.astype(float),
        "true_local_late": true_local_late,
    }


def fit_fuzzy_rd(
    y: np.ndarray,
    treatment: np.ndarray,
    running: np.ndarray,
    *,
    bandwidth: float = 0.75,
    bootstrap_reps: int = 300,
    seed: int = 999,
) -> FuzzyResult:
    """Estimate fuzzy RD as reduced-form jump divided by first-stage jump."""
    reduced_form = fit_local_linear(y, running, bandwidth=bandwidth)
    first_stage = fit_local_linear(treatment, running, bandwidth=bandwidth)
    if abs(first_stage.jump) < 1e-8:
        raise ValueError("first stage is too close to zero")
    wald = reduced_form.jump / first_stage.jump

    rng = np.random.default_rng(seed)
    n = y.size
    boot = []
    for _ in range(bootstrap_reps):
        idx = rng.integers(0, n, size=n)
        try:
            rf_b = fit_local_linear(y[idx], running[idx], bandwidth=bandwidth)
            fs_b = fit_local_linear(treatment[idx], running[idx], bandwidth=bandwidth)
            if abs(fs_b.jump) > 1e-8:
                boot.append(rf_b.jump / fs_b.jump)
        except ValueError:
            continue

    boot_arr = np.asarray(boot)
    if boot_arr.size < 20:
        bootstrap_se = float("nan")
        ci_low = float("nan")
        ci_high = float("nan")
    else:
        bootstrap_se = float(np.std(boot_arr, ddof=1))
        ci_low = float(np.percentile(boot_arr, 2.5))
        ci_high = float(np.percentile(boot_arr, 97.5))

    return FuzzyResult(
        bandwidth=bandwidth,
        reduced_form=reduced_form,
        first_stage=first_stage,
        wald=float(wald),
        bootstrap_se=bootstrap_se,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def global_difference(y: np.ndarray, treatment: np.ndarray) -> float:
    """Difference in mean outcomes between treated and untreated groups."""
    return float(np.mean(y[treatment == 1]) - np.mean(y[treatment == 0]))


def density_ratio(running: np.ndarray, *, cutoff: float = 0.0, window: float = 0.15) -> tuple[int, int, float]:
    """Count observations just below and just above the cutoff."""
    centered = running - cutoff
    left = int(np.sum((centered >= -window) & (centered < 0)))
    right = int(np.sum((centered >= 0) & (centered < window)))
    ratio = right / left if left > 0 else float("inf")
    return left, right, ratio


def simulate_manipulated_running(n: int = 4_000, seed: int = 456) -> np.ndarray:
    """Create a running variable with bunching just above the cutoff."""
    rng = np.random.default_rng(seed)
    running = rng.uniform(-2.0, 2.0, size=n)
    can_manipulate = (running > -0.20) & (running < 0.0)
    moved = can_manipulate & (rng.uniform(size=n) < 0.55)
    running[moved] = rng.uniform(0.0, 0.20, size=int(np.sum(moved)))
    return running


def binned_means(x: np.ndarray, y: np.ndarray, bins: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Return bin centers and mean y values for plotting."""
    edges = np.linspace(np.min(x), np.max(x), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(bins, np.nan)
    for j in range(bins):
        in_bin = (x >= edges[j]) & (x < edges[j + 1])
        if np.any(in_bin):
            means[j] = np.mean(y[in_bin])
    keep = np.isfinite(means)
    return centers[keep], means[keep]


def save_plots(
    output_dir: Path,
    sharp: dict[str, np.ndarray | float],
    fuzzy: dict[str, np.ndarray | float],
    sharp_fit: LocalLinearResult,
    fuzzy_result: FuzzyResult,
    manipulated_running: np.ndarray,
) -> None:
    """Save classroom figures for the simulated examples."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    running = np.asarray(sharp["running"])
    y = np.asarray(sharp["y"])
    centers, means = binned_means(running, y, bins=50)
    grid_left = np.linspace(-sharp_fit.bandwidth, 0, 100, endpoint=False)
    grid_right = np.linspace(0, sharp_fit.bandwidth, 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(centers, means, s=25, label="Binned means")
    ax.plot(grid_left, predict_local_linear(sharp_fit, grid_left), color="C1", label="Local linear fit")
    ax.plot(grid_right, predict_local_linear(sharp_fit, grid_right), color="C1")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Sharp RD: simulated cutoff")
    ax.set_xlabel("Running variable centered at cutoff")
    ax.set_ylabel("Outcome")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "sharp-rd-local-linear.png", dpi=160)
    plt.close(fig)

    running_f = np.asarray(fuzzy["running"])
    treatment_f = np.asarray(fuzzy["treatment"])
    centers, means = binned_means(running_f, treatment_f, bins=50)
    grid_left = np.linspace(-fuzzy_result.bandwidth, 0, 100, endpoint=False)
    grid_right = np.linspace(0, fuzzy_result.bandwidth, 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(centers, means, s=25, label="Binned treatment rate")
    ax.plot(grid_left, predict_local_linear(fuzzy_result.first_stage, grid_left), color="C2", label="First stage")
    ax.plot(grid_right, predict_local_linear(fuzzy_result.first_stage, grid_right), color="C2")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Fuzzy RD: treatment probability jumps")
    ax.set_xlabel("Running variable centered at cutoff")
    ax.set_ylabel("Treatment probability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fuzzy-rd-first-stage.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-0.6, 0.6, 41)
    ax.hist(np.asarray(sharp["running"]), bins=bins, alpha=0.6, label="No manipulation")
    ax.hist(manipulated_running, bins=bins, alpha=0.6, label="Manipulated")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Density diagnostic around the cutoff")
    ax.set_xlabel("Running variable centered at cutoff")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "density-diagnostic.png", dpi=160)
    plt.close(fig)


def print_sharp_results(sharp: dict[str, np.ndarray | float], bandwidths: list[float]) -> LocalLinearResult:
    running = np.asarray(sharp["running"])
    y = np.asarray(sharp["y"])
    d = np.asarray(sharp["treatment"])

    print("\nSharp RD simulation")
    print("-------------------")
    print(f"True treatment effect at the cutoff: {sharp['tau_at_cutoff']:.3f}")
    print(f"Global treated-control mean difference: {global_difference(y, d):.3f}")
    print("\nLocal linear estimates with triangular kernel weights:")
    print("bandwidth   n_eff  left right   tau_hat   robust_se        95% CI")

    last_result = None
    for h in bandwidths:
        result = fit_local_linear(y, running, bandwidth=h)
        last_result = result
        print(
            f"{h:9.2f} {result.n_eff:7d} {result.n_left:5d} {result.n_right:5d}"
            f" {result.jump:9.3f} {result.se:11.3f}"
            f" [{result.ci_low:7.3f}, {result.ci_high:7.3f}]"
        )

    covariate = np.asarray(sharp["covariate"])
    cov_jump = fit_local_linear(covariate, running, bandwidth=0.75)
    print("\nPredetermined covariate diagnostic:")
    print(
        f"Estimated covariate jump at h=0.75: {cov_jump.jump:.3f} "
        f"(robust se {cov_jump.se:.3f})"
    )

    if last_result is None:
        raise RuntimeError("no bandwidths supplied")
    return fit_local_linear(y, running, bandwidth=0.75)


def print_fuzzy_results(fuzzy: dict[str, np.ndarray | float], bandwidth: float, bootstrap_reps: int) -> FuzzyResult:
    running = np.asarray(fuzzy["running"])
    y = np.asarray(fuzzy["y"])
    treatment = np.asarray(fuzzy["treatment"])

    result = fit_fuzzy_rd(
        y,
        treatment,
        running,
        bandwidth=bandwidth,
        bootstrap_reps=bootstrap_reps,
    )

    print("\nFuzzy RD simulation")
    print("-------------------")
    print(f"Approximate true local complier effect: {fuzzy['true_local_late']:.3f}")
    print(f"Bandwidth: {bandwidth:.2f}")
    print(
        f"Reduced-form outcome jump: {result.reduced_form.jump:.3f} "
        f"(se {result.reduced_form.se:.3f})"
    )
    print(
        f"First-stage treatment jump: {result.first_stage.jump:.3f} "
        f"(se {result.first_stage.se:.3f})"
    )
    print(
        f"Local Wald estimate: {result.wald:.3f} "
        f"(bootstrap se {result.bootstrap_se:.3f}, "
        f"95% bootstrap CI [{result.ci_low:.3f}, {result.ci_high:.3f}])"
    )
    return result


def print_density_diagnostics(clean_running: np.ndarray, manipulated_running: np.ndarray) -> None:
    print("\nDensity/manipulation diagnostic")
    print("--------------------------------")
    for label, running in [("clean simulated running variable", clean_running), ("manipulated running variable", manipulated_running)]:
        left, right, ratio = density_ratio(running, window=0.15)
        print(f"{label}: count left={left}, right={right}, right/left={ratio:.2f}")
    print("A large right/left ratio near the cutoff is a warning sign, not a formal proof by itself.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate sharp and fuzzy regression discontinuity designs.")
    parser.add_argument("--n", type=int, default=4_000, help="Number of observations per simulation.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for the sharp RD simulation.")
    parser.add_argument("--bootstrap-reps", type=int, default=300, help="Bootstrap repetitions for fuzzy RD.")
    parser.add_argument("--save-plots", action="store_true", help="Save PNG figures for classroom use.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for plots. Defaults to a folder next to this script.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sharp = simulate_sharp_rd(n=args.n, seed=args.seed)
    fuzzy = simulate_fuzzy_rd(n=args.n, seed=args.seed + 198)
    manipulated = simulate_manipulated_running(n=args.n, seed=args.seed + 333)

    sharp_fit = print_sharp_results(sharp, bandwidths=[0.25, 0.50, 0.75, 1.25, 1.75])
    fuzzy_result = print_fuzzy_results(fuzzy, bandwidth=0.75, bootstrap_reps=args.bootstrap_reps)
    print_density_diagnostics(np.asarray(sharp["running"]), manipulated)

    if args.save_plots:
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = Path(__file__).with_name("lecture-9-rdd-figures")
        save_plots(output_dir, sharp, fuzzy, sharp_fit, fuzzy_result, manipulated)
        print(f"\nSaved plots to: {output_dir}")


if __name__ == "__main__":
    main()
