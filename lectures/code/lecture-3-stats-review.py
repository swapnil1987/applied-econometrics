"""Simulation examples for Applied Econometrics Lecture 3.

The script is self-contained: it generates synthetic school/test-score data
and randomized treatment-control data. It prints numerical summaries and can
display or save plots.

Run from the repository root:

    python lectures/code/lecture-3-stats-review.py
    python lectures/code/lecture-3-stats-review.py --save-plots
    python lectures/code/lecture-3-stats-review.py --no-plots
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def normal_cdf(x: float) -> float:
    """Standard normal CDF without requiring scipy."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_test(estimate: float, se: float, null: float = 0.0) -> dict[str, float]:
    """Large-sample two-sided z test and 95 percent confidence interval."""
    z = (estimate - null) / se
    p_value = 2.0 * (1.0 - normal_cdf(abs(z)))
    margin = 1.96 * se
    return {
        "estimate": estimate,
        "se": se,
        "z": z,
        "p_value": p_value,
        "ci_low": estimate - margin,
        "ci_high": estimate + margin,
    }


def simulate_school_rankings(
    rng: np.random.Generator,
    n_schools: int = 4_000,
    individual_sd: float = 12.0,
) -> dict[str, np.ndarray | float]:
    """Simulate school averages where school size has no causal effect."""
    # Right-skewed enrollment distribution: many small schools, fewer large ones.
    n_students = rng.lognormal(mean=4.35, sigma=0.85, size=n_schools).astype(int)
    n_students = np.clip(n_students, 10, 900)

    # True school quality is independent of size by construction.
    true_quality = rng.normal(loc=70.0, scale=2.0, size=n_schools)

    # The observed school average is the true school mean plus sampling noise.
    observed_avg = true_quality + rng.normal(
        loc=0.0,
        scale=individual_sd / np.sqrt(n_students),
        size=n_schools,
    )

    q_low, q_high = np.quantile(observed_avg, [0.01, 0.99])
    bottom = observed_avg <= q_low
    top = observed_avg >= q_high
    middle = ~(bottom | top)

    return {
        "n_students": n_students,
        "true_quality": true_quality,
        "observed_avg": observed_avg,
        "top": top,
        "bottom": bottom,
        "middle": middle,
        "top_mean_size": float(n_students[top].mean()),
        "bottom_mean_size": float(n_students[bottom].mean()),
        "middle_mean_size": float(n_students[middle].mean()),
        "top_true_quality": float(true_quality[top].mean()),
        "bottom_true_quality": float(true_quality[bottom].mean()),
        "middle_true_quality": float(true_quality[middle].mean()),
    }


def simulate_experiment_once(
    rng: np.random.Generator,
    n_treated: int,
    n_control: int,
    tau: float = 4.0,
    baseline_mean: float = 70.0,
    outcome_sd: float = 12.0,
) -> dict[str, float]:
    """Simulate one randomized treatment-control study."""
    control = rng.normal(loc=baseline_mean, scale=outcome_sd, size=n_control)
    treated = rng.normal(loc=baseline_mean + tau, scale=outcome_sd, size=n_treated)

    estimate = float(treated.mean() - control.mean())
    s_treat = float(treated.std(ddof=1))
    s_control = float(control.std(ddof=1))
    se = math.sqrt(s_treat**2 / n_treated + s_control**2 / n_control)
    result = normal_test(estimate, se)
    result["n_treated"] = float(n_treated)
    result["n_control"] = float(n_control)
    return result


def simulate_many_experiments(
    rng: np.random.Generator,
    reps: int,
    n_treated: int,
    n_control: int,
    tau: float = 4.0,
    outcome_sd: float = 12.0,
) -> np.ndarray:
    """Repeated experiments to show the sampling distribution of ATE estimates."""
    estimates = np.empty(reps)
    for r in range(reps):
        result = simulate_experiment_once(
            rng,
            n_treated=n_treated,
            n_control=n_control,
            tau=tau,
            outcome_sd=outcome_sd,
        )
        estimates[r] = result["estimate"]
    return estimates


def ci_coverage_experiment(
    rng: np.random.Generator,
    reps: int = 5_000,
    n_treated: int = 75,
    n_control: int = 75,
    tau: float = 4.0,
) -> dict[str, float]:
    """Estimate repeated-sample coverage of nominal 95 percent CIs."""
    covers = 0
    rejects = 0
    ci_widths = []
    for _ in range(reps):
        result = simulate_experiment_once(rng, n_treated, n_control, tau=tau)
        covers += result["ci_low"] <= tau <= result["ci_high"]
        rejects += result["p_value"] < 0.05
        ci_widths.append(result["ci_high"] - result["ci_low"])

    return {
        "coverage": covers / reps,
        "rejection_rate_against_zero": rejects / reps,
        "mean_ci_width": float(np.mean(ci_widths)),
    }


def print_school_summary(school: dict[str, np.ndarray | float]) -> None:
    n_students = school["n_students"]
    observed_avg = school["observed_avg"]
    assert isinstance(n_students, np.ndarray)
    assert isinstance(observed_avg, np.ndarray)

    corr = float(np.corrcoef(n_students, observed_avg)[0, 1])
    print("\n=== School ranking simulation ===")
    print("School size has no causal effect in the data generating process.")
    print(f"Number of schools: {len(n_students):,}")
    print(f"Median school size: {np.median(n_students):.0f}")
    print(f"Mean size among top 1% observed scores: {school['top_mean_size']:.1f}")
    print(f"Mean size among middle 98% observed scores: {school['middle_mean_size']:.1f}")
    print(f"Mean size among bottom 1% observed scores: {school['bottom_mean_size']:.1f}")
    print(f"Correlation(size, observed average): {corr:.3f}")
    print("Mean true quality by observed-score group:")
    print(f"  top 1%:    {school['top_true_quality']:.2f}")
    print(f"  middle:    {school['middle_true_quality']:.2f}")
    print(f"  bottom 1%: {school['bottom_true_quality']:.2f}")


def print_experiment_summary(rng: np.random.Generator) -> None:
    print("\n=== One randomized treatment-control study ===")
    for n in [25, 100, 400]:
        result = simulate_experiment_once(rng, n_treated=n, n_control=n)
        print(
            f"n per group={n:>3}: "
            f"estimate={result['estimate']:>6.2f}, "
            f"SE={result['se']:>5.2f}, "
            f"95% CI=[{result['ci_low']:>6.2f}, {result['ci_high']:>6.2f}], "
            f"p={result['p_value']:.3f}"
        )

    print("\n=== Repeated-study behavior ===")
    for n in [25, 100, 400]:
        estimates = simulate_many_experiments(
            rng,
            reps=5_000,
            n_treated=n,
            n_control=n,
        )
        empirical_sd = estimates.std(ddof=1)
        theoretical_se = math.sqrt(12.0**2 / n + 12.0**2 / n)
        wrong_sign = np.mean(estimates < 0.0)
        print(
            f"n per group={n:>3}: "
            f"SD(estimates)={empirical_sd:>5.2f}, "
            f"formula SE={theoretical_se:>5.2f}, "
            f"P(estimate < 0)={wrong_sign:.3f}"
        )

    coverage = ci_coverage_experiment(rng)
    print("\n=== Repeated 95% CI experiment ===")
    print(f"Empirical coverage of true tau=4: {coverage['coverage']:.3f}")
    print(f"Power against zero at 5% level: {coverage['rejection_rate_against_zero']:.3f}")
    print(f"Average CI width: {coverage['mean_ci_width']:.2f}")


def make_plots(
    school: dict[str, np.ndarray | float],
    rng: np.random.Generator,
    save_plots: bool,
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib is not installed; skipping plots.")
        return

    n_students = school["n_students"]
    observed_avg = school["observed_avg"]
    true_quality = school["true_quality"]
    top = school["top"]
    bottom = school["bottom"]
    middle = school["middle"]
    assert isinstance(n_students, np.ndarray)
    assert isinstance(observed_avg, np.ndarray)
    assert isinstance(true_quality, np.ndarray)
    assert isinstance(top, np.ndarray)
    assert isinstance(bottom, np.ndarray)
    assert isinstance(middle, np.ndarray)

    figures: list[tuple[str, object]] = []

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(n_students[middle], observed_avg[middle], s=10, alpha=0.18, label="Middle 98%")
    ax.scatter(n_students[top], observed_avg[top], s=28, alpha=0.85, label="Top 1%")
    ax.scatter(n_students[bottom], observed_avg[bottom], s=28, alpha=0.85, label="Bottom 1%")
    ax.set_xscale("log")
    ax.set_xlabel("Number of tested students (log scale)")
    ax.set_ylabel("Observed average score")
    ax.set_title("Small schools populate both tails of observed rankings")
    ax.legend()
    figures.append(("lecture-3-school-extremes.png", fig))

    fig, ax = plt.subplots(figsize=(9, 5))
    order = np.argsort(n_students)
    ax.plot(n_students[order], true_quality[order], ".", alpha=0.18, label="True school mean")
    ax.plot(n_students[order], observed_avg[order], ".", alpha=0.18, label="Observed average")
    ax.set_xscale("log")
    ax.set_xlabel("Number of tested students (log scale)")
    ax.set_ylabel("Score")
    ax.set_title("Observed averages are much noisier for small schools")
    ax.legend()
    figures.append(("lecture-3-true-vs-observed.png", fig))

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(-8, 16, 60)
    for n, color in [(25, "C0"), (100, "C1"), (400, "C2")]:
        estimates = simulate_many_experiments(
            rng,
            reps=5_000,
            n_treated=n,
            n_control=n,
        )
        ax.hist(
            estimates,
            bins=bins,
            density=True,
            alpha=0.35,
            color=color,
            label=f"n={n} per group",
        )
    ax.axvline(4.0, color="black", linestyle="--", linewidth=1.5, label="True effect")
    ax.axvline(0.0, color="gray", linestyle=":", linewidth=1.5, label="Zero")
    ax.set_xlabel("Estimated treatment effect")
    ax.set_ylabel("Density")
    ax.set_title("Sampling distribution shrinks as sample size grows")
    ax.legend()
    figures.append(("lecture-3-treatment-estimates.png", fig))

    if save_plots:
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename, fig in figures:
            path = output_dir / filename
            fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"\nSaved {len(figures)} plots to {output_dir}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=20260429,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only print numerical output.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots instead of opening an interactive window.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "lecture-3-figures",
        help="Directory used with --save-plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    school = simulate_school_rankings(rng)
    print_school_summary(school)
    print_experiment_summary(rng)

    if not args.no_plots:
        make_plots(
            school=school,
            rng=rng,
            save_plots=args.save_plots,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
