"""Port of ``10-algorithm-compare.jl`` — fit all ten regressors, rank them by RMSD.

The Julia version built a ``Dict`` of model types, ran

    RMS = [ _fit((X, y, xtest, ytest), key) for key in models_keys ]
    for (k, v) in zip(models_keys, RMS); @info "$(k) => $(v)"; end

Here :func:`reg10.models.score_models` does the fitting loop and the result is a tidy
``DataFrame`` (printed as a table and saved to ``figures/compare-rmsd.csv``) plus two
figures.

Run:  ``python scripts/compare_algorithms.py``  (``--keys Lasso Ridge`` to subset)
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _common import FIGURE_DIR, load_split, use_backend  # noqa: E402

MODEL_ORDER = [
    "Linear",
    "Huber",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "SGD",
    "SVR",
    "NuSVR",
    "MLP",
    "RandomForest",
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keys", nargs="*", default=MODEL_ORDER, choices=MODEL_ORDER)
    parser.add_argument("--show", action="store_true", help="open figures instead of saving only")
    parser.add_argument("--quiet", action="store_true", help="do not print each RMSD while fitting")
    args = parser.parse_args(argv)

    use_backend(args.show)
    from reg10 import plots
    from reg10.models import coefficient_table, comparison_table, model_collection, score_models

    df, (X_train, X_test, y_train, y_test) = load_split()
    print(f"{df.shape[0]} rows | {X_train.shape[0]} train / {X_test.shape[0]} test")

    specs = model_collection()
    fitted = score_models(
        X_train, y_train, X_test, y_test, args.keys, specs=specs, verbose=not args.quiet
    )
    table = comparison_table(fitted)
    print("\n" + table.to_string(index=False))

    coefs = coefficient_table(fitted)
    print("\nslopes, USD per unit of feature:\n" + coefs.round(2).to_string())

    FIGURE_DIR.mkdir(exist_ok=True)
    plots.rmsd_barplot(table).savefig(FIGURE_DIR / "compare-rmsd.png", dpi=130)
    plots.coefficient_heatmap(coefs).savefig(FIGURE_DIR / "compare-coefficients.png", dpi=130)
    table.to_csv(FIGURE_DIR / "compare-rmsd.csv", index=False)
    print(f"\nfigures + csv written to {FIGURE_DIR}/")
    if args.show:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
