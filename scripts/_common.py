"""Shared scaffolding for the ported per-algorithm scripts.

Mirrors how the Julia scripts all began with ``include("data-processing.jl")``: load the
frame, split 70/30, fit one model, print the MLJ-style report, save the GLMakie-style
figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"

# Make ``src/reg10`` importable without installing the package (mirrors Julia's include()).
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402


def use_backend(show: bool = False) -> None:
    """Headless by default so the scripts run in CI; ``--show`` opens a window."""
    if not show:
        matplotlib.use("Agg")


def load_split():
    """``df = get_data(); y, X = unpack(...); partition((X, y), 0.7)``."""
    from reg10.data_processing import TARGET, get_data, partition, unpack

    df = get_data()
    X, y = unpack(df, TARGET)
    return df, partition(X, y)


def report_and_plot(key: str, *, show: bool = False, tag: str | None = None) -> None:
    """Fit one model from the registry, print its report, and write its figures."""
    from reg10 import plots
    from reg10.models import fit_one, model_collection

    use_backend(show)
    tag = tag or key
    spec = model_collection()[key]
    _, (X_train, X_test, y_train, y_test) = load_split()
    fitted = fit_one(spec, X_train, y_train, X_test, y_test)

    print(f"--- {key} ({spec.julia_model} in MLJ, pkg {spec.julia_pkg}) ---")
    print("fitted params :", fitted.params)
    print(f"price at feature means : ${fitted.prediction_at_mean:,.1f}")
    print("coefficients (USD per unit of feature):")
    print(fitted.coefficients().round(2).to_string())
    print(f"RMSD  train {fitted.rmsd_train:,.1f}   test {fitted.rmsd_test:,.1f}"
          f"   (fit {fitted.fit_seconds:.3f}s)")

    FIGURE_DIR.mkdir(exist_ok=True)
    plots.residue_plot(fitted.residuals, title=f"{key} — test residuals").savefig(
        FIGURE_DIR / f"{tag}-residues.png", dpi=120
    )
    plots.residual_hist(fitted.residuals, title=f"{key} — residual distribution").savefig(
        FIGURE_DIR / f"{tag}-residual-hist.png", dpi=120
    )
    plots.predicted_vs_actual(fitted.y_true, fitted.y_pred, title=f"{key} — predicted vs actual").savefig(
        FIGURE_DIR / f"{tag}-pred-vs-actual.png", dpi=120
    )
    print(f"figures written to {FIGURE_DIR}/{tag}-*.png")
    if show:
        matplotlib.pyplot.show()
