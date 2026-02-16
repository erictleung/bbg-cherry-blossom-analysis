import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    https://www.pymc.io/projects/examples/en/latest/survival_analysis/survival_analysis.html
    https://austinrochford.com/posts/revisit-survival-pymc.html
    """)
    return


@app.cell
def _():
    import arviz as az
    import lifelines as ll
    from matplotlib import pyplot as plt
    from matplotlib.ticker import NullLocator
    import numpy as np
    import nutpie #
    import pandas as pd
    import pymc as pm #
    from pymc.distributions.timeseries import GaussianRandomWalk
    import seaborn as sns #
    from pytensor import tensor as pt #
    import pytensor
    from pytensor import tensor as T
    from seaborn import objects as so
    from statsmodels.datasets import get_rdataset
    return T, az, np, pd, plt, pm, pytensor, sns


@app.cell
def _(pytensor):
    pytensor.config.blas__ldflags = '-llapack -lblas -lcblas'
    return


@app.cell
def _(az, np):
    RANDOM_SEED = 8927
    rng = np.random.default_rng(RANDOM_SEED)
    az.style.use("arviz-darkgrid")
    return (RANDOM_SEED,)


@app.cell
def _(plt, sns):
    plt.rc("figure", figsize=(8, 6))
    sns.set(color_codes=True)
    return


@app.cell
def _(np, pd, pm):
    try:
        df = pd.read_csv("../data/mastectomy.csv")
    except FileNotFoundError:
        df = pd.read_csv(pm.get_data("mastectomy.csv"))

    df.event = df.event.astype(np.int64)
    df.metastasized = (df.metastasized == "yes").astype(np.int64)
    n_patients = df.shape[0]
    patients = np.arange(n_patients)
    return df, n_patients, patients


@app.cell
def _(df):
    df
    return


@app.cell
def _(mo):
    mo.md(r"""
    There are 42 observations with three variables, with survival times in months after mastectomy of women with breast cancer. Cancer is classified as having metastized or not.
    """)
    return


@app.cell
def _(df):
    # Percent of observations observed
    df.event.mean()
    # I.e., 40% of observations censored
    return


@app.cell
def _(df, n_patients, patients, plt):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hlines(
        patients[df.event.values == 0], 0, df[df.event.values == 0].time, color="C3", label="Censored"
    )

    ax.hlines(
        patients[df.event.values == 1], 0, df[df.event.values == 1].time, color="C7", label="Uncensored"
    )

    ax.scatter(
        df[df.metastasized.values == 1].time,
        patients[df.metastasized.values == 1],
        color="k",
        zorder=10,
        label="Metastasized",
    )

    ax.set_xlim(left=0)
    ax.set_xlabel("Months since mastectomy")
    ax.set_yticks([])
    ax.set_ylabel("Subject")

    ax.set_ylim(-0.25, n_patients + 0.25)

    ax.legend(loc="center right");

    plt.show()
    return


@app.cell
def _(df, np):
    interval_length = 3  # in months
    interval_bounds = np.arange(0, df.time.max() + interval_length + 1, interval_length)
    n_intervals = interval_bounds.size - 1
    intervals = np.arange(n_intervals)
    return interval_bounds, interval_length, intervals, n_intervals


@app.cell
def _(df, interval_bounds, plt):
    def _():
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.hist(
            df[df.event == 0].time.values,
            bins=interval_bounds,
            lw=0,
            color="C3",
            alpha=0.5,
            label="Censored",
        )

        ax.hist(
            df[df.event == 1].time.values,
            bins=interval_bounds,
            lw=0,
            color="C7",
            alpha=0.5,
            label="Uncensored",
        )

        ax.set_xlim(0, interval_bounds[-1])
        ax.set_xlabel("Months since mastectomy")

        ax.set_yticks([0, 1, 2, 3])
        ax.set_ylabel("Number of observations")

        ax.legend()
        return plt.show()

    _()
    return


@app.cell
def _(df, interval_length, n_intervals, n_patients, np, patients):
    last_period = np.floor((df.time - 0.01) / interval_length).astype(int)

    death = np.zeros((n_patients, n_intervals))
    death[patients, last_period] = df.event
    return death, last_period


@app.cell
def _(df, interval_bounds, interval_length, last_period, np, patients):
    exposure = np.greater_equal.outer(df.time.to_numpy(), interval_bounds[:-1]) * interval_length
    exposure[patients, last_period] = df.time - interval_bounds[last_period]
    return (exposure,)


@app.cell
def _(exposure):
    exposure
    return


@app.cell
def _(T, death, df, exposure, intervals, pm):
    coords = {"intervals": intervals}

    with pm.Model(coords=coords) as model:
        lambda0 = pm.Gamma("lambda0", 0.01, 0.01, dims="intervals")

        beta = pm.Normal("beta", 0, sigma=1000)

        lambda_ = pm.Deterministic("lambda_", T.outer(T.exp(beta * df.metastasized), lambda0))
        mu = pm.Deterministic("mu", exposure * lambda_)

        obs = pm.Poisson("obs", mu, observed=death)
    return (model,)


@app.cell
def _():
    n_samples = 1000
    n_tune = 1000
    return n_samples, n_tune


@app.cell
def _(RANDOM_SEED, model, n_samples, n_tune, pm):
    with model:
        idata = pm.sample(
            n_samples,
            tune=n_tune,
            target_accept=0.99,
            random_seed=RANDOM_SEED,
        )
    return (idata,)


@app.cell
def _(idata, np):
    np.exp(idata.posterior["beta"]).mean()
    return


if __name__ == "__main__":
    app.run()
