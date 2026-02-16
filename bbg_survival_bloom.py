import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


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
    from pytensor import tensor as pt #
    import seaborn as sns #
    from seaborn import objects as so
    from statsmodels.datasets import get_rdataset
    return (pd,)


@app.cell
def _(pd):
    df = pd.read_csv("https://raw.githubusercontent.com/erictleung/bbggplots/main/data-raw/bbg_tree_bloom_2025.csv")
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    # Categorize specifically for peak bloom
    df['peak_bloom'] = df['bloom'] == 'Peak Bloom'
    return


@app.cell
def _(df):
    # Keep only earliest date
    (
        df
        .query('peak_bloom == True')
        .groupby(['id', 'tree'])
        ['date']
        .min()
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
