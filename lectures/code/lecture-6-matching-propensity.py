"""Lecture 6 example: matching and propensity-score weighting.

The script simulates an observational study with selection on observables.
It compares:

1. the naive treated-control difference,
2. a regression-adjusted estimate,
3. nearest-neighbor matching for the ATT,
4. propensity-score IPW for the ATE,
5. propensity-score weighting for the ATT.

No external data are downloaded. Only NumPy is required.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def add_intercept(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(x.shape[0]), x])


def standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    sd = x.std(axis=0, ddof=1)
    sd = np.where(sd == 0.0, 1.0, sd)
    return (x - mean) / sd, mean, sd


def ols_coef(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return beta


def fit_logit_irls(
    d: np.ndarray,
    x: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-9,
    ridge: float = 1e-8,
) -> np.ndarray:
    """Fit a simple logistic regression by Newton/IRLS.

    This avoids depending on scikit-learn or statsmodels. The tiny ridge term
    only stabilizes the linear solve; it is not used on the intercept.
    """

    beta = np.zeros(x.shape[1])
    penalty = np.eye(x.shape[1]) * ridge
    penalty[0, 0] = 0.0

    for _ in range(max_iter):
        p = sigmoid(x @ beta)
        w = np.clip(p * (1.0 - p), 1e-8, None)
        grad = x.T @ (d - p) - penalty @ beta
        hess = (x.T * w) @ x + penalty
        step = np.linalg.solve(hess, grad)
        beta_new = beta + step
        if np.max(np.abs(step)) < tol:
            beta = beta_new
            break
        beta = beta_new

    return beta


@dataclass
class SimulatedData:
    y: np.ndarray
    d: np.ndarray
    x: np.ndarray
    y0: np.ndarray
    y1: np.ndarray
    tau: np.ndarray
    covariate_names: tuple[str, ...]
    true_pscore: np.ndarray


def simulate_data(n: int, seed: int) -> SimulatedData:
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(45.0, 12.0, n), 20.0, 75.0)
    baseline_score = rng.normal(0.0, 1.0, n)
    income = rng.lognormal(mean=10.35, sigma=0.45, size=n)
    urban = rng.binomial(1, 0.55, n).astype(float)

    age_z = (age - age.mean()) / age.std(ddof=1)
    income_z = (income - income.mean()) / income.std(ddof=1)

    true_logit = (
        -0.30
        + 0.70 * baseline_score
        + 0.55 * income_z
        + 0.45 * urban
        - 0.25 * age_z
    )
    true_pscore = sigmoid(true_logit)
    d = rng.binomial(1, true_pscore, n).astype(float)

    eps = rng.normal(0.0, 4.0, n)
    y0 = (
        35.0
        + 3.2 * baseline_score
        + 0.00018 * income
        + 1.8 * urban
        - 0.06 * age
        + 1.2 * np.sin(baseline_score)
        + eps
    )
    tau = 5.0 + 1.4 * urban + 0.8 * baseline_score
    y1 = y0 + tau
    y = d * y1 + (1.0 - d) * y0

    x = np.column_stack([age, baseline_score, income, urban])
    names = ("age", "baseline_score", "income", "urban")
    return SimulatedData(y, d, x, y0, y1, tau, names, true_pscore)


def regression_adjusted_ate(data: SimulatedData) -> float:
    x_std, _, _ = standardize(data.x)
    design = add_intercept(np.column_stack([data.d, x_std]))
    beta = ols_coef(data.y, design)
    return float(beta[1])


def nearest_neighbor_att(data: SimulatedData, k: int = 1) -> tuple[float, dict[str, float]]:
    x_std, _, _ = standardize(data.x)
    treated = np.flatnonzero(data.d == 1.0)
    controls = np.flatnonzero(data.d == 0.0)
    x_controls = x_std[controls]
    y_controls = data.y[controls]

    imputed_y0 = np.empty(treated.size)
    nearest_distances = np.empty(treated.size)
    matched_controls = []

    for pos, idx in enumerate(treated):
        diff = x_controls - x_std[idx]
        dist2 = np.einsum("ij,ij->i", diff, diff)
        nn = np.argpartition(dist2, k - 1)[:k]
        imputed_y0[pos] = y_controls[nn].mean()
        nearest_distances[pos] = np.sqrt(dist2[nn]).mean()
        matched_controls.extend(controls[nn].tolist())

    att = float(np.mean(data.y[treated] - imputed_y0))
    unique_controls = len(set(matched_controls))
    diagnostics = {
        "mean_neighbor_distance": float(nearest_distances.mean()),
        "p90_neighbor_distance": float(np.quantile(nearest_distances, 0.90)),
        "unique_controls_used": float(unique_controls),
        "treated_units": float(treated.size),
    }
    return att, diagnostics


def propensity_scores(data: SimulatedData) -> np.ndarray:
    x_std, _, _ = standardize(data.x)
    # Include a few simple nonlinear terms to improve balance.
    age_z, base_z, income_z, urban = x_std.T
    features = np.column_stack(
        [
            age_z,
            base_z,
            income_z,
            urban,
            base_z**2,
            income_z**2,
            base_z * income_z,
            urban * base_z,
        ]
    )
    design = add_intercept(features)
    beta = fit_logit_irls(data.d, design)
    return np.clip(sigmoid(design @ beta), 0.01, 0.99)


def ipw_ate(y: np.ndarray, d: np.ndarray, ehat: np.ndarray) -> float:
    treated = d == 1.0
    controls = ~treated
    mu1 = np.sum(y[treated] / ehat[treated]) / np.sum(1.0 / ehat[treated])
    mu0 = np.sum(y[controls] / (1.0 - ehat[controls])) / np.sum(
        1.0 / (1.0 - ehat[controls])
    )
    return float(mu1 - mu0)


def ipw_att(y: np.ndarray, d: np.ndarray, ehat: np.ndarray) -> float:
    treated = d == 1.0
    controls = ~treated
    control_w = ehat[controls] / (1.0 - ehat[controls])
    mu1_treated = y[treated].mean()
    mu0_treated = np.sum(control_w * y[controls]) / np.sum(control_w)
    return float(mu1_treated - mu0_treated)


def effective_sample_size(w: np.ndarray) -> float:
    return float((w.sum() ** 2) / np.sum(w**2))


def standardized_mean_differences(
    x: np.ndarray,
    d: np.ndarray,
    names: tuple[str, ...],
    weights: np.ndarray | None = None,
) -> list[tuple[str, float]]:
    treated = d == 1.0
    controls = ~treated
    out: list[tuple[str, float]] = []

    for j, name in enumerate(names):
        pooled_sd = x[:, j].std(ddof=1)
        if weights is None:
            mean_t = x[treated, j].mean()
            mean_c = x[controls, j].mean()
        else:
            wt = weights[treated]
            wc = weights[controls]
            mean_t = np.average(x[treated, j], weights=wt)
            mean_c = np.average(x[controls, j], weights=wc)
        out.append((name, float((mean_t - mean_c) / pooled_sd)))

    return out


def common_support_mask(d: np.ndarray, ehat: np.ndarray) -> np.ndarray:
    treated = d == 1.0
    controls = ~treated
    lower = max(ehat[treated].min(), ehat[controls].min())
    upper = min(ehat[treated].max(), ehat[controls].max())
    return (ehat >= lower) & (ehat <= upper)


def print_table(rows: list[tuple[str, str, float, float | None]]) -> None:
    print("\nEstimator comparison")
    print("-" * 79)
    print(f"{'Method':<38} {'Target':<12} {'Estimate':>10} {'Bias vs target':>14}")
    print("-" * 79)
    for method, target, estimate, bias in rows:
        bias_text = "" if bias is None else f"{bias:14.3f}"
        print(f"{method:<38} {target:<12} {estimate:10.3f} {bias_text}")
    print("-" * 79)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3000, help="number of observations")
    parser.add_argument("--seed", type=int, default=20260429, help="random seed")
    args = parser.parse_args()

    data = simulate_data(args.n, args.seed)
    ehat = propensity_scores(data)

    true_ate = float(np.mean(data.tau))
    true_att = float(np.mean(data.tau[data.d == 1.0]))

    naive = float(data.y[data.d == 1.0].mean() - data.y[data.d == 0.0].mean())
    reg_adj = regression_adjusted_ate(data)
    nn_att, match_diag = nearest_neighbor_att(data, k=1)
    ps_ate = ipw_ate(data.y, data.d, ehat)
    ps_att = ipw_att(data.y, data.d, ehat)

    support = common_support_mask(data.d, ehat)
    ps_ate_trim = ipw_ate(data.y[support], data.d[support], ehat[support])
    ps_att_trim = ipw_att(data.y[support], data.d[support], ehat[support])

    rows = [
        ("Naive treated-control gap", "not causal", naive, None),
        ("Regression adjusted OLS", "ATE", reg_adj, reg_adj - true_ate),
        ("1-NN covariate matching", "ATT", nn_att, nn_att - true_att),
        ("Propensity-score IPW", "ATE", ps_ate, ps_ate - true_ate),
        ("Propensity-score ATT weights", "ATT", ps_att, ps_att - true_att),
        ("Trimmed propensity-score IPW", "ATE trimmed", ps_ate_trim, None),
        ("Trimmed propensity-score ATT", "ATT trimmed", ps_att_trim, None),
    ]

    print(f"Simulated observations: {args.n}")
    print(f"Treated share: {data.d.mean():.3f}")
    print(f"True ATE: {true_ate:.3f}")
    print(f"True ATT: {true_att:.3f}")
    print_table(rows)

    treated = data.d == 1.0
    controls = ~treated
    print("\nEstimated propensity-score overlap")
    print("-" * 79)
    print(
        f"Treated scores:  min={ehat[treated].min():.3f}, "
        f"p50={np.median(ehat[treated]):.3f}, max={ehat[treated].max():.3f}"
    )
    print(
        f"Control scores:  min={ehat[controls].min():.3f}, "
        f"p50={np.median(ehat[controls]):.3f}, max={ehat[controls].max():.3f}"
    )
    print(
        f"Common-support trimming keeps {support.sum()} of {args.n} observations "
        f"({support.mean():.1%})."
    )

    ate_weights = data.d / ehat + (1.0 - data.d) / (1.0 - ehat)
    att_weights = np.where(data.d == 1.0, 1.0, ehat / (1.0 - ehat))
    print("\nWeight diagnostics")
    print("-" * 79)
    print(f"ATE weights: max={ate_weights.max():.2f}, ESS={effective_sample_size(ate_weights):.1f}")
    print(f"ATT weights: max={att_weights.max():.2f}, ESS={effective_sample_size(att_weights):.1f}")

    print("\nNearest-neighbor matching diagnostics")
    print("-" * 79)
    print(f"Mean standardized neighbor distance: {match_diag['mean_neighbor_distance']:.3f}")
    print(f"90th percentile neighbor distance: {match_diag['p90_neighbor_distance']:.3f}")
    print(
        "Unique controls used: "
        f"{int(match_diag['unique_controls_used'])} for "
        f"{int(match_diag['treated_units'])} treated units"
    )

    print("\nStandardized mean differences")
    print("-" * 79)
    print(f"{'Covariate':<20} {'Raw':>10} {'ATE weighted':>15}")
    weighted_balance = standardized_mean_differences(data.x, data.d, data.covariate_names, ate_weights)
    raw_balance = standardized_mean_differences(data.x, data.d, data.covariate_names)
    for (name, raw), (_, weighted) in zip(raw_balance, weighted_balance):
        print(f"{name:<20} {raw:10.3f} {weighted:15.3f}")

    print("\nTeaching notes")
    print("-" * 79)
    print("The naive gap is biased because treatment depends on observed covariates.")
    print("Matching targets the treated population in this implementation.")
    print("IPW can target either ATE or ATT, but it becomes fragile with extreme scores.")


if __name__ == "__main__":
    main()
