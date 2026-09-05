"""Matplotlib figures that reproduce the GLMakie plots of the Julia project.

============================  =========================================  ========================
Julia (GLMakie)               Python (matplotlib)                        this module
============================  =========================================  ========================
``cor-plot.jl`` ``plot_cor``  pairs grid: ``density!`` + ``scatter!``     :func:`pairs_plot`
``heatmap!`` + ``text!``      ``ax.imshow`` + ``ax.annotate``             :func:`cor_heatmap`
``stem!`` + ``hlines!``       ``ax.stem`` + ``ax.axhline``                :func:`residue_plot`
``scatter(ytest, ŷ)``         ``ax.scatter`` + identity line              :func:`predicted_vs_actual`
``hist(res, density=true)``   ``ax.hist(density=True)`` + KDE             :func:`residual_hist`
``@info`` RMSD loop           ranked bar chart                            :func:`rmsd_barplot`
============================  =========================================  ========================

All helpers return the :class:`matplotlib.figure.Figure` so that Quarto can capture them
(the Julia code ended each cell with a bare ``fig`` for the same reason).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt

#: Julia ``colors = [:orange, :lightgreen, :purple, :lightblue, :red, :green]``
PAIR_COLORS = ["tab:orange", "tab:green", "tab:purple", "tab:blue", "tab:red", "tab:green"]


def _finish(fig: plt.Figure) -> plt.Figure:
    fig.tight_layout()
    return fig


def pairs_plot(
    df: pd.DataFrame,
    *,
    cols: list[str] | None = None,
    alpha: float = 0.45,
    size: float = 6,
    max_points: int = 1500,
    random_state: int = 123,
) -> plt.Figure:
    """Scatter/ density matrix -- port of ``plot_cor(df)`` from ``cor-plot.jl``.

    Diagonal: kernel density of each variable (Makie ``density!``).
    Off-diagonal: scatter of row variable against column variable (Makie ``scatter!``).

    ``max_points`` subsamples the scatter: the Julia figure drew all 5,000 rows in every
    one of the 30 panels, which makes a very heavy PNG for a web page.
    """
    frame = df if cols is None else df[cols]
    if max_points is not None and len(frame) > max_points:
        frame = frame.sample(max_points, random_state=random_state)
    n = frame.shape[1]
    fig, axes = plt.subplots(n, n, figsize=(size * n / 2, size * n / 2))
    axes = np.atleast_2d(axes)
    values = {c: frame[c].to_numpy(float) for c in frame.columns}

    for i, row in enumerate(frame.columns):
        for j, col in enumerate(frame.columns):
            ax = axes[i, j]
            color = PAIR_COLORS[j % len(PAIR_COLORS)]
            if i == j:
                x = values[col]
                grid = np.linspace(x.min(), x.max(), 300)
                ax.plot(grid, gaussian_kde(x)(grid), color=color, lw=1.5)
                ax.fill_between(grid, gaussian_kde(x)(grid), color=color, alpha=0.35)
                ax.set_yticks([])
            else:
                ax.scatter(values[col], values[row], s=4, alpha=alpha, color=color, linewidths=0)
            if j > 0:
                ax.set_yticklabels([])
            if i < n - 1:
                ax.set_xticklabels([])
            ax.tick_params(labelsize=7)

    # add_xy_label(): column names along the bottom, variable names down the left
    for i, name in enumerate(frame.columns):
        axes[-1, i].set_xlabel(name, fontsize=9)
        axes[i, 0].set_ylabel(name, fontsize=9)
    fig.suptitle("Pairwise structure of the USA housing data", y=1.0)
    return _finish(fig)


def cor_heatmap(df: pd.DataFrame, *, digits: int = 3, cmap: str = "viridis") -> plt.Figure:
    """Correlation matrix with annotated cells -- the ``heatmap!`` cell in the Julia note."""
    corr = df.corr(numeric_only=True).round(digits)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(corr.to_numpy(float), cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    for x in range(corr.shape[1]):
        for y in range(corr.shape[0]):
            v = corr.to_numpy()[y, x]
            ax.annotate(
                f"{v:.{digits}f}",
                (x, y),
                ha="center",
                va="center",
                fontsize=9,
                color="red" if x == y else "white",
            )
    ax.set_title("House price correlation matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _finish(fig)


def residue_plot(
    residuals,
    *,
    title: str = "",
    size=(12, 5),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """``stem(res)`` with the red ``hlines!(ax, [0])`` zero reference -- ``plot_residue``."""
    own_ax = ax is None
    if own_ax:
        fig, ax = plt.subplots(figsize=size)
    else:
        fig = ax.figure
    res = np.asarray(residuals, dtype=float)
    markerline, stemlines, baseline = ax.stem(np.arange(res.size), res, basefmt=" ")
    markerline.set_markersize(3)
    plt.setp(stemlines, lw=0.8, color="tab:blue")
    ax.axhline(0, color="red", alpha=0.5, lw=1.5)
    ax.set_xlabel("test row")
    ax.set_ylabel("residual (USD)")
    ax.set_xlim(0, max(res.size - 1, 1))
    if title:
        ax.set_title(title)
    return _finish(fig)


def predicted_vs_actual(y_true, y_pred, *, title: str = "", size=(6, 6)) -> plt.Figure:
    """``scatter(ytest, ŷ)`` plus the y = x reference."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    fig, ax = plt.subplots(figsize=size)
    ax.scatter(y_true, y_pred, s=8, alpha=0.5, color="tab:blue", linewidths=0)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], color="red", alpha=0.6, lw=1.5)
    ax.set_xlabel("actual price (USD)")
    ax.set_ylabel("predicted price (USD)")
    ax.set_title(title or "Predicted vs actual")
    return _finish(fig)


def residual_hist(residuals, *, bins: int = 60, title: str = "", size=(8, 4.5)) -> plt.Figure:
    """``hist(res, density=true)`` with a fitted Gaussian overlay for scale reference."""
    res = np.asarray(residuals, dtype=float)
    fig, ax = plt.subplots(figsize=size)
    ax.hist(res, bins=bins, density=True, color="tab:orange", alpha=0.65)
    grid = np.linspace(res.min(), res.max(), 400)
    ax.plot(grid, gaussian_kde(res)(grid), color="tab:red", lw=1.5, label="KDE")
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("residual (USD)")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    ax.set_title(title or "Residual distribution")
    return _finish(fig)


def hist(df: pd.Series, *, bins: int = 60, color: str = "tab:blue", label: str = "", title: str = "") -> plt.Figure:
    """The Julia ``hist(df[!, :Price])`` cells."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df, bins=bins, color=color, alpha=0.7)
    ax.set_xlabel(label or df.name)
    ax.set_ylabel("count")
    ax.set_title(title or f"Distribution of {label or df.name}")
    return _finish(fig)


def scatter(x: pd.Series, y: pd.Series, *, size=(7, 5), alpha: float = 0.5, color: str = "tab:purple") -> plt.Figure:
    """The ``scatter!(x1, x2)`` cells (single predictor against the target)."""
    fig, ax = plt.subplots(figsize=size)
    ax.scatter(x, y, s=10, alpha=alpha, color=color, linewidths=0)
    ax.set_xlabel(x.name)
    ax.set_ylabel(y.name)
    return _finish(fig)


def residue_grid(fitted, *, ncols: int = 2, size=(13, 4.2)) -> plt.Figure:
    """Small multiples of the test residuals for several fitted models.

    The Julia note drew one ``stem!`` figure per algorithm; stacking them keeps the
    comparison readable.
    """
    n = len(list(fitted))
    fitted = list(fitted)
    nrows = int(np.ceil(len(fitted) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size[0], size[1] * nrows / 2), squeeze=False)
    for ax, f in zip(axes.ravel(), fitted):
        res = np.asarray(f.residuals, dtype=float)
        markerline, stemlines, _ = ax.stem(np.arange(res.size), res, basefmt=" ")
        markerline.set_markersize(2)
        plt.setp(stemlines, lw=0.5, color="tab:blue")
        ax.axhline(0, color="red", alpha=0.5, lw=1)
        ax.set_title(f"{f.key}  (RMSD {f.rmsd_test:,.0f})", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(fitted) :]:
        ax.axis("off")
    fig.suptitle("Test-set residuals: predicted minus actual price", y=1.0)
    return _finish(fig)


def parity_grid(fitted, *, ncols: int = 2, size=(9, 4.2)) -> plt.Figure:
    """Small multiples of ``scatter(ytest, ŷ)`` -- one panel per fitted model."""
    fitted = list(fitted)
    nrows = int(np.ceil(len(fitted) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size[0], size[1] * nrows / 2), squeeze=False)
    for ax, f in zip(axes.ravel(), fitted):
        y_true = np.asarray(f.y_true, dtype=float)
        y_pred = np.asarray(f.y_pred, dtype=float)
        ax.scatter(y_true, y_pred, s=4, alpha=0.4, linewidths=0, color="tab:blue")
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], color="red", alpha=0.6, lw=1)
        ax.set_title(f.key, fontsize=9)
        ax.set_xlabel("actual", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(fitted) :]:
        ax.axis("off")
    fig.suptitle("Predicted vs actual price", y=1.0)
    return _finish(fig)


def rmsd_barplot(table: pd.DataFrame, *, value_col: str = "RMSD (test)", model_col: str = "model") -> plt.Figure:
    """Bar chart of the comparison table (the Julia script only printed these numbers)."""
    tab = table.sort_values(value_col)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(tab[model_col], tab[value_col], color="tab:blue", alpha=0.8)
    ax.bar_label(
        bars,
        fmt="%.0f",
        padding=2,
        fontsize=8,
    )
    ax.set_ylabel(f"{value_col} (USD) — lower is better")
    ax.set_title("Ten regression algorithms, ranked by test RMSD")
    ax.margins(y=0.12)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _finish(fig)


def coefficient_heatmap(coefs: pd.DataFrame) -> plt.Figure:
    """Signed local slopes (dollar units) for every model x feature.

    The colour scale is symmetric around zero and clipped at the 90th percentile of
    |slope|, so that a few huge tree-model differences do not wash out the rest.
    """
    values = coefs.to_numpy(float)
    limit = float(np.percentile(np.abs(values[np.isfinite(values)]), 90)) or 1.0
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(values, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(coefs.shape[1]), coefs.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(coefs.shape[0]), coefs.index, fontsize=8)
    for x in range(coefs.shape[1]):
        for y in range(coefs.shape[0]):
            v = values[y, x]
            ax.annotate(f"{v/1e3:,.0f}k", (x, y), ha="center", va="center", fontsize=7)
    ax.set_title("Local sensitivity of fitted price to each feature (per unit of X)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    return _finish(fig)
