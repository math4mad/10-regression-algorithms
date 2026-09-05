"""Port of ``NuSVR-regressor.jl`` — nu-SVR with an RBF kernel on the USA housing data.

Run:  ``python scripts/nu_svr.py``   (add ``--show`` to open the figures instead of only saving)
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _common import report_and_plot  # noqa: E402

if __name__ == "__main__":
    report_and_plot("NuSVR", show="--show" in sys.argv, tag="NuSVR-regressor")
