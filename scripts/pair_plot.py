"""Port of ``cor-plot.jl`` — pairwise scatter/density grid and the correlation heatmap.

The Julia function ``plot_cor(df)`` built a GLMakie ``Figure(cols, cols)`` with
``density!`` on the diagonal, ``scatter!`` off the diagonal and axis labels on the outer
edges; the Quarto note then added a ``heatmap!`` of ``cor(df)`` with annotated cells.
Both live in :mod:`reg10.plots`; this script just writes them to files.

Run:  ``python scripts/pair_plot.py``  (``--show`` to open the windows)
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _common import FIGURE_DIR, load_split, use_backend  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--dpi", type=int, default=130)
    args = parser.parse_args(argv)

    use_backend(args.show)
    from reg10 import plots

    df, _ = load_split()
    FIGURE_DIR.mkdir(exist_ok=True)

    pairs = plots.pairs_plot(df)
    pairs.savefig(FIGURE_DIR / "us-housing-cor.png", dpi=args.dpi)  # Julia: us-housing-cor.png
    heatmap = plots.cor_heatmap(df)
    heatmap.savefig(FIGURE_DIR / "us-housing-corr-heatmap.png", dpi=args.dpi)
    print(f"wrote {FIGURE_DIR}/us-housing-cor.png and us-housing-corr-heatmap.png")
    print("correlation matrix:\n" + df.corr(numeric_only=True).round(3).to_string())

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
