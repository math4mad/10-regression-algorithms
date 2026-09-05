"""Port of ``elastic-net-regressor.jl`` — Elastic Net regression on the USA housing data.

Run:  ``python scripts/elastic_net_regressor.py``   (add ``--show`` to open the figures instead of only saving)
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _common import report_and_plot  # noqa: E402

if __name__ == "__main__":
    report_and_plot("ElasticNet", show="--show" in sys.argv, tag="elastic-net-regressor")
