"""
Lecture 10 simulation: basic macroeconometrics.

This script uses simulated data only. It demonstrates:
- AR(1) persistence and forecasts;
- spurious regression with unrelated random walks;
- a small VAR(1) and impulse responses.

The script writes no files unless --save-figures is supplied.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class OLSResult:
    beta: np.ndarray
    se: np.ndarray
    resid: np.ndarray
    r2: float


@dataclass
class AR1Result:
    c: float
    phi: float
    se_c: float
    se_phi: float
    mean: float
    forecasts: np.ndarray


@dataclass
class VAR1Result:
    intercept: np.ndarray
    transition: np.ndarray
    residual_cov: np.ndarray
    impulse_responses: np.ndarray


def ols(y: np.ndarray, x: np.ndarray) -> OLSResult:
    """Plain OLS with homoskedastic standard errors."""
    beta = np.linalg.pinv(x.T @ x) @ (x.T @ y)
    resid = y - x @ beta
    n_obs, n_par = x.shape
    sigma2 = float(resid.T @ resid / max(n_obs - n_par, 1))
    cov = sigma2 * np.linalg.pinv(x.T @ x)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    centered = y - y.mean()
    r2 = 1.0 - float(resid.T @ resid / (centered.T @ centered))
    return OLSResult(beta=beta, se=se, resid=resid, r2=r2)


def newey_west_se(y: np.ndarray, x: np.ndarray, lags: int) -> np.ndarray:
    """Heteroskedasticity-and-autocorrelation consistent standard errors."""
    beta = np.linalg.pinv(x.T @ x) @ (x.T @ y)
    resid = y - x @ beta
    n_obs = x.shape[0]
    bread = np.linalg.pinv(x.T @ x)

    meat = np.zeros((x.shape[1], x.shape[1]))
    xu = x * resid[:, None]
    meat += xu.T @ xu

    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = xu[lag:].T @ xu[:-lag]
        meat += weight * (gamma + gamma.T)

    cov = bread @ meat @ bread
    cov *= n_obs / max(n_obs - x.shape[1], 1)
    return np.sqrt(np.maximum(np.diag(cov), 0.0))


def simulate_ar1(n: int = 160, c: float = 0.4, phi: float = 0.82, sigma: float = 1.0, seed: int = 123) -> np.ndarray:
    """Simulate a stationary AR(1)."""
    rng = np.random.default_rng(seed)
    y = np.empty(n)
    y[0] = c / (1.0 - phi)
    shocks = rng.normal(0.0, sigma, size=n)
    for t in range(1, n):
        y[t] = c + phi * y[t - 1] + shocks[t]
    return y


def estimate_ar1(y: np.ndarray, horizon: int = 12) -> AR1Result:
    """Estimate AR(1) and produce h-step forecasts from the final observation."""
    y_now = y[1:]
    y_lag = y[:-1]
    x = np.column_stack([np.ones_like(y_lag), y_lag])
    fit = ols(y_now, x)
    c_hat, phi_hat = fit.beta
    mean_hat = c_hat / (1.0 - phi_hat) if abs(phi_hat) < 0.999 else np.nan

    forecasts = []
    current = y[-1]
    for h in range(1, horizon + 1):
        if np.isfinite(mean_hat):
            forecasts.append(mean_hat + phi_hat**h * (current - mean_hat))
        else:
            forecasts.append(current + h * c_hat)

    return AR1Result(
        c=float(c_hat),
        phi=float(phi_hat),
        se_c=float(fit.se[0]),
        se_phi=float(fit.se[1]),
        mean=float(mean_hat),
        forecasts=np.asarray(forecasts),
    )


def simulate_random_walks(n: int = 180, seed: int = 456) -> tuple[np.ndarray, np.ndarray]:
    """Simulate two unrelated random walks."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0.0, 1.0, size=n))
    y = np.cumsum(rng.normal(0.0, 1.0, size=n))
    return y, x


def spurious_regression(y: np.ndarray, x: np.ndarray) -> tuple[OLSResult, np.ndarray]:
    """Regress one unrelated random walk on another."""
    design = np.column_stack([np.ones_like(x), x])
    fit = ols(y, design)
    hac_se = newey_west_se(y, design, lags=8)
    return fit, hac_se


def simulate_var1(n: int = 220, seed: int = 789) -> np.ndarray:
    """Simulate a stable two-variable VAR(1)."""
    rng = np.random.default_rng(seed)
    intercept = np.array([0.10, 0.05])
    transition = np.array(
        [
            [0.55, -0.18],
            [0.20, 0.62],
        ]
    )
    structural_cov = np.array(
        [
            [1.00, 0.45],
            [0.45, 0.80],
        ]
    )
    chol = np.linalg.cholesky(structural_cov)
    z = np.zeros((n, 2))
    for t in range(1, n):
        shock = chol @ rng.normal(size=2)
        z[t] = intercept + transition @ z[t - 1] + shock
    return z


def estimate_var1(z: np.ndarray, horizon: int = 12, shock_index: int = 0) -> VAR1Result:
    """Estimate a VAR(1) and compute Cholesky impulse responses."""
    y = z[1:]
    x = np.column_stack([np.ones(z.shape[0] - 1), z[:-1]])
    beta = np.linalg.pinv(x.T @ x) @ (x.T @ y)
    resid = y - x @ beta
    intercept = beta[0]
    transition = beta[1:].T
    residual_cov = resid.T @ resid / max(resid.shape[0] - x.shape[1], 1)

    chol = np.linalg.cholesky(residual_cov)
    shock = chol[:, shock_index]
    responses = np.empty((horizon + 1, z.shape[1]))
    power = np.eye(z.shape[1])
    for h in range(horizon + 1):
        responses[h] = power @ shock
        power = transition @ power

    return VAR1Result(
        intercept=intercept,
        transition=transition,
        residual_cov=residual_cov,
        impulse_responses=responses,
    )


def autocorrelation(y: np.ndarray, max_lag: int = 8) -> np.ndarray:
    """Sample autocorrelations for lags 1,...,max_lag."""
    y_centered = y - y.mean()
    denom = float(y_centered @ y_centered)
    return np.array(
        [
            float(y_centered[lag:] @ y_centered[:-lag] / denom)
            for lag in range(1, max_lag + 1)
        ]
    )


def print_ar_summary(y: np.ndarray, result: AR1Result) -> None:
    print("=" * 78)
    print("AR(1): persistence and forecasting")
    print("-" * 78)
    print(f"Estimated c:                 {result.c: .3f}  (se {result.se_c: .3f})")
    print(f"Estimated phi:               {result.phi: .3f}  (se {result.se_phi: .3f})")
    print(f"Implied stationary mean:      {result.mean: .3f}")
    print(f"Final observed value:         {y[-1]: .3f}")
    print("Sample autocorrelations, lags 1-8:")
    print(np.array2string(autocorrelation(y), precision=3, suppress_small=True))
    print("Forecasts for horizons 1-6:")
    print(np.array2string(result.forecasts[:6], precision=3, suppress_small=True))
    print()


def print_spurious_summary(fit: OLSResult, hac_se: np.ndarray) -> None:
    beta = fit.beta
    naive_t = beta[1] / fit.se[1]
    hac_t = beta[1] / hac_se[1]
    print("=" * 78)
    print("Spurious regression: unrelated random walks")
    print("-" * 78)
    print("Regression: y_t = alpha + beta x_t + u_t")
    print(f"Estimated beta:              {beta[1]: .3f}")
    print(f"Naive standard error:         {fit.se[1]: .3f}")
    print(f"Naive t-statistic:            {naive_t: .3f}")
    print(f"Newey-West standard error:    {hac_se[1]: .3f}")
    print(f"Newey-West t-statistic:       {hac_t: .3f}")
    print(f"R-squared:                    {fit.r2: .3f}")
    print("The data-generating relationship is zero; the impressive fit is accidental.")
    print()


def print_var_summary(result: VAR1Result) -> None:
    print("=" * 78)
    print("VAR(1): joint dynamics and impulse responses")
    print("-" * 78)
    print("Estimated intercept:")
    print(np.array2string(result.intercept, precision=3, suppress_small=True))
    print("Estimated transition matrix A:")
    print(np.array2string(result.transition, precision=3, suppress_small=True))
    print("Reduced-form residual covariance:")
    print(np.array2string(result.residual_cov, precision=3, suppress_small=True))
    print("Cholesky impulse response to shock 1, horizons 0-6:")
    print(np.array2string(result.impulse_responses[:7], precision=3, suppress_small=True))
    print()


def make_plots(
    ar_series: np.ndarray,
    random_walk_y: np.ndarray,
    random_walk_x: np.ndarray,
    var_result: VAR1Result,
    *,
    save_figures: bool,
    show: bool,
) -> None:
    """Create simple plots if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plots.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(ar_series, color="tab:blue")
    axes[0, 0].set_title("Stationary AR(1)")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("y")

    axes[0, 1].plot(random_walk_y, label="y random walk")
    axes[0, 1].plot(random_walk_x, label="x random walk")
    axes[0, 1].set_title("Unrelated random walks")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].legend()

    axes[1, 0].scatter(random_walk_x, random_walk_y, s=14, alpha=0.7)
    axes[1, 0].set_title("Spurious levels relationship")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")

    horizons = np.arange(var_result.impulse_responses.shape[0])
    axes[1, 1].axhline(0.0, color="black", linewidth=1)
    axes[1, 1].plot(horizons, var_result.impulse_responses[:, 0], marker="o", label="variable 1")
    axes[1, 1].plot(horizons, var_result.impulse_responses[:, 1], marker="o", label="variable 2")
    axes[1, 1].set_title("VAR impulse response")
    axes[1, 1].set_xlabel("Horizon")
    axes[1, 1].legend()

    fig.tight_layout()

    if save_figures:
        out = Path(__file__).with_name("lecture-10-macroeconometrics-simulation.png")
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

    ar_series = simulate_ar1()
    ar_result = estimate_ar1(ar_series)
    print_ar_summary(ar_series, ar_result)

    random_walk_y, random_walk_x = simulate_random_walks()
    spurious_fit, spurious_hac = spurious_regression(random_walk_y, random_walk_x)
    print_spurious_summary(spurious_fit, spurious_hac)

    var_series = simulate_var1()
    var_result = estimate_var1(var_series)
    print_var_summary(var_result)

    make_plots(
        ar_series,
        random_walk_y,
        random_walk_x,
        var_result,
        save_figures=args.save_figures,
        show=args.show,
    )


if __name__ == "__main__":
    main()
