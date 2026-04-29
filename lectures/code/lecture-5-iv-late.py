"""
Lecture 5 examples: instrumental variables, weak instruments, and LATE.

The script is self-contained: it simulates all data and uses only NumPy.
Run from the course root with:

    python lectures/code/lecture-5-iv-late.py

The examples are intentionally small and transparent so students can inspect
the data-generating process and compare the estimators to known truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RegressionResult:
    beta: np.ndarray
    se: np.ndarray
    residuals: np.ndarray
    r2: float


def add_constant(*cols: np.ndarray) -> np.ndarray:
    """Build a regression matrix with an intercept and one or more columns."""
    n = len(cols[0])
    pieces = [np.ones(n)]
    for col in cols:
        arr = np.asarray(col)
        if arr.ndim == 1:
            pieces.append(arr)
        else:
            pieces.extend(arr.T)
    return np.column_stack(pieces)


def ols(y: np.ndarray, x: np.ndarray) -> RegressionResult:
    """Ordinary least squares with homoskedastic standard errors."""
    y = np.asarray(y)
    x = np.asarray(x)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ beta
    n, k = x.shape
    sigma2 = residuals @ residuals / (n - k)
    xtx_inv = np.linalg.pinv(x.T @ x)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    tss = ((y - y.mean()) @ (y - y.mean()))
    r2 = 1.0 - (residuals @ residuals) / tss
    return RegressionResult(beta=beta, se=se, residuals=residuals, r2=r2)


def tsls(y: np.ndarray, d: np.ndarray, z: np.ndarray) -> RegressionResult:
    """
    Just-identified 2SLS for one endogenous treatment and one excluded
    instrument, with an intercept.
    """
    y = np.asarray(y)
    w = add_constant(d)  # structural regressors: intercept and treatment
    instruments = add_constant(z)

    ztz_inv = np.linalg.pinv(instruments.T @ instruments)
    a = w.T @ instruments @ ztz_inv @ instruments.T @ w
    b = w.T @ instruments @ ztz_inv @ instruments.T @ y
    beta = np.linalg.pinv(a) @ b

    residuals = y - w @ beta
    n, k = w.shape
    sigma2 = residuals @ residuals / (n - k)
    se = np.sqrt(np.diag(sigma2 * np.linalg.pinv(a)))
    tss = ((y - y.mean()) @ (y - y.mean()))
    r2 = 1.0 - (residuals @ residuals) / tss
    return RegressionResult(beta=beta, se=se, residuals=residuals, r2=r2)


def mean_diff(y: np.ndarray, z: np.ndarray) -> float:
    """Difference in means, E[Y|Z=1] - E[Y|Z=0]."""
    return float(y[z == 1].mean() - y[z == 0].mean())


def first_stage_f(d: np.ndarray, z: np.ndarray) -> float:
    """For one binary instrument, the first-stage F equals the squared t-stat."""
    fs = ols(d, add_constant(z))
    return float((fs.beta[1] / fs.se[1]) ** 2)


def print_rows(title: str, rows: list[tuple[str, float | str]]) -> None:
    print()
    print(title)
    print("-" * len(title))
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        if isinstance(value, str):
            text = value
        else:
            text = f"{value: .4f}"
        print(f"{name:<{width}} : {text}")


def simulate_endogeneity(
    rng: np.random.Generator,
    n: int = 50_000,
    first_stage_strength: float = 0.8,
) -> dict[str, np.ndarray]:
    """
    Treatment is endogenous because it depends on an omitted variable U.
    The instrument Z is randomly assigned and shifts D.
    """
    u = rng.normal(0.0, 1.0, n)
    z = rng.binomial(1, 0.5, n)
    v = rng.normal(0.0, 1.0, n)
    e = rng.normal(0.0, 1.0, n)

    d = first_stage_strength * z + 0.7 * u + v
    y = 2.0 * d + 1.5 * u + e
    return {"y": y, "d": d, "z": z, "u": u}


def example_endogeneity_and_iv(rng: np.random.Generator) -> None:
    data = simulate_endogeneity(rng)
    y, d, z, u = data["y"], data["d"], data["z"], data["u"]

    naive_ols = ols(y, add_constant(d))
    oracle_ols = ols(y, add_constant(d, u))
    first_stage = ols(d, add_constant(z))
    reduced_form = ols(y, add_constant(z))
    wald = mean_diff(y, z) / mean_diff(d, z)
    iv = tsls(y, d, z)

    print_rows(
        "1. Endogeneity and IV recovery",
        [
            ("True treatment effect", 2.0),
            ("Naive OLS, omits U", naive_ols.beta[1]),
            ("Oracle OLS, controls U", oracle_ols.beta[1]),
            ("First-stage slope", first_stage.beta[1]),
            ("First-stage F-stat", first_stage_f(d, z)),
            ("Reduced-form slope", reduced_form.beta[1]),
            ("Wald IV", wald),
            ("2SLS", iv.beta[1]),
            ("2SLS standard error", iv.se[1]),
        ],
    )


def example_weak_instruments(rng: np.random.Generator) -> None:
    strengths = [1.00, 0.50, 0.25, 0.10, 0.05]
    reps = 250
    n = 2_500

    rows: list[tuple[str, float | str]] = []
    for strength in strengths:
        iv_estimates = []
        f_stats = []
        for _ in range(reps):
            data = simulate_endogeneity(
                rng,
                n=n,
                first_stage_strength=strength,
            )
            y, d, z = data["y"], data["d"], data["z"]
            denom = mean_diff(d, z)
            if abs(denom) < 1e-10:
                continue
            iv_estimates.append(mean_diff(y, z) / denom)
            f_stats.append(first_stage_f(d, z))

        iv_arr = np.asarray(iv_estimates)
        f_arr = np.asarray(f_stats)
        rows.extend(
            [
                (f"strength {strength:0.2f}: median F", float(np.median(f_arr))),
                (f"strength {strength:0.2f}: IV median", float(np.median(iv_arr))),
                (f"strength {strength:0.2f}: IV p05", float(np.percentile(iv_arr, 5))),
                (f"strength {strength:0.2f}: IV p95", float(np.percentile(iv_arr, 95))),
                (f"strength {strength:0.2f}: IV sd", float(iv_arr.std(ddof=1))),
            ]
        )

    print_rows("2. Weak instruments: same true effect, less first-stage information", rows)


def simulate_noncompliance(rng: np.random.Generator, n: int = 80_000) -> dict[str, np.ndarray]:
    """
    Random assignment Z is an instrument for treatment received D.

    Compliance type is correlated with baseline outcome and treatment effects.
    That makes treatment received endogenous even though assignment is random.
    """
    z = rng.binomial(1, 0.5, n)
    latent = rng.normal(0.0, 1.0, n)
    noise = rng.normal(0.0, 2.0, n)

    # Type shares are roughly: 30 percent never, 55 percent complier,
    # 15 percent always. Types depend on latent income/motivation.
    compliance_type = np.full(n, "complier", dtype=object)
    compliance_type[latent < -0.52] = "never"
    compliance_type[latent > 1.04] = "always"

    d0 = np.zeros(n)
    d1 = np.zeros(n)
    d1[compliance_type == "complier"] = 1
    d0[compliance_type == "always"] = 1
    d1[compliance_type == "always"] = 1

    d = np.where(z == 1, d1, d0)

    # Heterogeneous treatment effects. The IV estimand should recover the
    # average effect for compliers, not the population average.
    tau = np.zeros(n)
    tau[compliance_type == "never"] = 2.0
    tau[compliance_type == "complier"] = 6.0
    tau[compliance_type == "always"] = 10.0

    y0 = 50.0 + 8.0 * latent + noise
    y = y0 + tau * d + rng.normal(0.0, 1.0, n)
    return {
        "y": y,
        "d": d,
        "z": z,
        "tau": tau,
        "type": compliance_type,
    }


def example_noncompliance_late(rng: np.random.Generator) -> None:
    data = simulate_noncompliance(rng)
    y, d, z = data["y"], data["d"], data["z"]
    tau, compliance_type = data["tau"], data["type"]

    itt_y = mean_diff(y, z)
    itt_d = mean_diff(d, z)
    wald = itt_y / itt_d
    naive_received = mean_diff(y, d)
    iv = tsls(y, d, z)

    rows: list[tuple[str, float | str]] = [
        ("Share never-taker", float(np.mean(compliance_type == "never"))),
        ("Share complier", float(np.mean(compliance_type == "complier"))),
        ("Share always-taker", float(np.mean(compliance_type == "always"))),
        ("Observed E[D|Z=0]", float(d[z == 0].mean())),
        ("Observed E[D|Z=1]", float(d[z == 1].mean())),
        ("First stage / complier share", itt_d),
        ("True population ATE", float(tau.mean())),
        ("True complier LATE", float(tau[compliance_type == "complier"].mean())),
        ("Naive received-treatment diff", naive_received),
        ("ITT effect of assignment", itt_y),
        ("Wald LATE", wald),
        ("2SLS LATE", iv.beta[1]),
    ]
    print_rows("3. Non-compliance and LATE", rows)


def main() -> None:
    rng = np.random.default_rng(20260429)
    example_endogeneity_and_iv(rng)
    example_weak_instruments(rng)
    example_noncompliance_late(rng)


if __name__ == "__main__":
    main()
