"""
Self-contained examples for Applied Econometrics Lecture 4.

Topics:
1. Omitted-variable bias and regression anatomy.
2. Dummy variables and grouped regression.
3. Bad controls: mediators and colliders.

The script uses simulated data only. It deliberately avoids downloaded data and
statsmodels, so the algebra behind each OLS coefficient is visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ols(y, X, names, weights=None):
    """Return a compact OLS/WLS table using matrix formulas."""
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    X = np.asarray(X, dtype=float)

    if weights is None:
        Xw = X
        yw = y
    else:
        w = np.asarray(weights, dtype=float).reshape(-1, 1)
        Xw = X * np.sqrt(w)
        yw = y * np.sqrt(w)

    beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    resid = y - X @ beta

    n, k = X.shape
    if weights is None:
        sigma2 = float(resid.T @ resid / (n - k))
        vcov = sigma2 * np.linalg.inv(X.T @ X)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1, 1)
        # Frequency-weight style variance. Coefficients are the object of
        # interest in the lecture; standard errors are only illustrative here.
        sigma2 = float((w * resid).T @ resid / (w.sum() - k))
        vcov = sigma2 * np.linalg.inv(X.T @ (w * X))

    se = np.sqrt(np.diag(vcov)).reshape(-1, 1)
    out = pd.DataFrame(
        {
            "coef": beta.ravel(),
            "std_err": se.ravel(),
        },
        index=names,
    )
    return out


def add_constant(*cols):
    n = len(cols[0])
    return np.column_stack([np.ones(n), *cols])


def slope(y, x):
    y = np.asarray(y)
    x = np.asarray(x)
    return np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)


def print_section(title):
    bar = "=" * len(title)
    print(f"\n{title}\n{bar}")


def example_omitted_variable_bias(seed=123, n=4000):
    rng = np.random.default_rng(seed)

    ability = rng.normal(0, 1, n)
    family_background = rng.normal(0, 1, n)
    motivation = rng.normal(0, 1, n)

    # Education is partly chosen by latent ability and family background.
    education = (
        12
        + 1.1 * ability
        + 0.7 * family_background
        + 0.5 * motivation
        + rng.normal(0, 1.5, n)
    )

    true_return = 0.08
    log_wage = (
        1.8
        + true_return * education
        + 0.35 * ability
        + 0.20 * family_background
        + rng.normal(0, 0.35, n)
    )

    # Imperfect proxy observed by the econometrician.
    test_score = 100 + 12 * ability + rng.normal(0, 8, n)
    parent_index = 0.7 * family_background + rng.normal(0, 0.8, n)

    short = ols(
        log_wage,
        add_constant(education),
        ["constant", "education"],
    )
    proxy_controls = ols(
        log_wage,
        add_constant(education, test_score, parent_index),
        ["constant", "education", "test_score", "parent_index"],
    )
    long_oracle = ols(
        log_wage,
        add_constant(education, ability, family_background),
        ["constant", "education", "ability", "family_background"],
    )

    # FWL: residualize education on observed controls, then regress Y on the
    # residualized education. This recovers the multiple-regression coefficient.
    controls = add_constant(test_score, parent_index)
    aux = np.linalg.lstsq(controls, education, rcond=None)[0]
    educ_tilde = education - controls @ aux
    fwl_coef = slope(log_wage, educ_tilde)

    print_section("Example 1: Omitted-variable bias and FWL")
    print(f"True causal return to one year of education: {true_return:.3f}")
    print("\nShort regression, omitting ability and family background:")
    print(short.round(4))
    print("\nAdd imperfect observed proxies:")
    print(proxy_controls.round(4))
    print("\nOracle regression with latent confounders:")
    print(long_oracle.round(4))
    print(
        "\nFWL coefficient using residualized education: "
        f"{fwl_coef:.4f} "
        "(matches education coefficient with proxies)"
    )


def example_dummy_and_grouped(seed=456, n=5000):
    rng = np.random.default_rng(seed)

    ability = rng.normal(0, 1, n)
    education = np.clip(np.round(12 + 1.2 * ability + rng.normal(0, 2, n)), 8, 18)
    education = education.astype(int)

    # Nonlinear wage schedule by years of education.
    schedule = {
        year: 14 + 0.9 * (year - 8) + 0.18 * max(year - 12, 0) ** 2
        for year in range(8, 19)
    }
    wage = np.array([schedule[e] for e in education]) + 2.0 * ability + rng.normal(0, 4, n)

    linear = ols(wage, add_constant(education), ["constant", "education"])

    levels = sorted(np.unique(education))
    base = levels[0]
    dummy_cols = [(education == level).astype(float) for level in levels[1:]]
    dummy_names = [f"educ_{level}_minus_{base}" for level in levels[1:]]
    dummy_fit = ols(wage, add_constant(*dummy_cols), ["constant", *dummy_names])

    df = pd.DataFrame({"wage": wage, "education": education})
    group = (
        df.assign(count=1)
        .groupby("education", as_index=False)
        .agg(wage=("wage", "mean"), count=("count", "sum"))
    )

    grouped_weighted = ols(
        group["wage"],
        add_constant(group["education"]),
        ["constant", "education"],
        weights=group["count"],
    )
    grouped_unweighted = ols(
        group["wage"],
        add_constant(group["education"]),
        ["constant", "education"],
    )

    fitted_dummy_means = {}
    for level in levels:
        if level == base:
            fitted_dummy_means[level] = float(dummy_fit.loc["constant", "coef"])
        else:
            fitted_dummy_means[level] = float(
                dummy_fit.loc["constant", "coef"]
                + dummy_fit.loc[f"educ_{level}_minus_{base}", "coef"]
            )

    mean_check = group.copy()
    mean_check["dummy_fitted_mean"] = mean_check["education"].map(fitted_dummy_means)
    mean_check["difference"] = mean_check["wage"] - mean_check["dummy_fitted_mean"]

    print_section("Example 2: Dummy variables and grouped regression")
    print("\nLinear regression forces one slope:")
    print(linear.round(4))
    print("\nDummy regression leaves education fully flexible.")
    print("First six dummy coefficients:")
    print(dummy_fit.head(7).round(4))
    print("\nCell means equal fitted values from saturated dummy regression:")
    print(mean_check.head(8).round(4))
    print("\nIndividual OLS and weighted grouped OLS match for Y on education:")
    print(pd.concat(
        {
            "individual": linear["coef"],
            "grouped_weighted": grouped_weighted["coef"],
            "grouped_unweighted": grouped_unweighted["coef"],
        },
        axis=1,
    ).round(4))


def example_bad_controls(seed=789, n=8000):
    rng = np.random.default_rng(seed)

    treatment = rng.binomial(1, 0.5, n)
    responsiveness = rng.normal(0, 1, n)

    # Mediator: treatment makes agreement more likely; agreement raises payment.
    agreement_latent = -0.3 + 1.0 * treatment + 1.0 * responsiveness + rng.normal(0, 1, n)
    agreement = (agreement_latent > 0).astype(float)
    payment = 100 + 4 * treatment + 18 * agreement + 5 * responsiveness + rng.normal(0, 8, n)

    total_effect = ols(payment, add_constant(treatment), ["constant", "treatment"])
    with_mediator = ols(
        payment,
        add_constant(treatment, agreement),
        ["constant", "treatment", "agreement"],
    )

    # Collider/common-effect control. Treatment is randomized, but the control
    # is caused by treatment and by an unobserved determinant of the outcome.
    u = rng.normal(0, 1, n)
    y = 50 + 6 * treatment + 10 * u + rng.normal(0, 3, n)
    collider = 0.9 * treatment + 0.9 * u + rng.normal(0, 1, n)

    randomized = ols(y, add_constant(treatment), ["constant", "treatment"])
    collider_control = ols(
        y,
        add_constant(treatment, collider),
        ["constant", "treatment", "collider"],
    )

    print_section("Example 3: Bad controls")
    print("\nRandomized treatment with a mediator.")
    print("The first model estimates the total effect of the email-like treatment.")
    print(total_effect.round(4))
    print("\nAdding the mediator holds part of the causal path fixed.")
    print(with_mediator.round(4))
    print("\nRandomized treatment with a collider/common-effect control.")
    print("Without the collider, randomization works:")
    print(randomized.round(4))
    print("\nControlling for the collider induces bias:")
    print(collider_control.round(4))


def main():
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 8)
    example_omitted_variable_bias()
    example_dummy_and_grouped()
    example_bad_controls()


if __name__ == "__main__":
    main()
