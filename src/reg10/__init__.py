"""reg10 — Python port of the Julia "Practical Introduction to 10 Regression Algorithms".

Modules
-------
data_processing : port of ``data-processing.jl``
models          : port of ``10-algorithm-compare.jl`` (the 10-estimator registry)
plots           : port of ``cor-plot.jl`` + the GLMakie residue/scatter/hist figures
"""

from .data_processing import FEATURES, TARGET, get_data, load_data, unpack, partition
from .models import MODEL_KEYS, model_collection, rmsd, score_models

__all__ = [
    "FEATURES",
    "TARGET",
    "unpack",
    "partition",
    "get_data",
    "load_data",
    "MODEL_KEYS",
    "model_collection",
    "rmsd",
    "score_models",
]
