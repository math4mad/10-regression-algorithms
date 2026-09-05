"""The ten regressors, and the RMSD comparison harness.

Port of ``10-algorithm-compare.jl`` (``LinearMolesCollections()`` plus the ``_fit``
list-comprehension and the ``@info`` printing loop).  Every Julia model has a
scikit-learn counterpart:

=========================  =====================================  ==========================
Julia (MLJ model)          Python (scikit-learn)                  Julia package
=========================  =====================================  ==========================
``LinearRegressor``        ``LinearRegression``                   MLJLinearModels
``RobustRegressor``        ``HuberRegressor``                     MLJLinearModels
``RidgeRegressor``         ``Ridge``                              MLJLinearModels
``LassoRegressor``         ``Lasso``                              MLJLinearModels
``ElasticNetRegressor``    ``ElasticNet``                         MLJLinearModels
``SGDRegressor``           ``SGDRegressor``                       MLJScikitLearnInterface
``EpsilonSVR``             ``SVR(kernel="poly")``                 LIBSVM
``NuSVR``                  ``NuSVR``                              LIBSVM
``NeuralNetworkRegressor`` ``MLPRegressor``                       MLJFlux / Flux
``RandomForestRegressor``  ``RandomForestRegressor``              DecisionTree
=========================  =====================================  ==========================

Design note -- scaling
----------------------
The Julia scripts fitted directly on raw columns (area income ~1e5, price ~1e6).
Several scikit-learn estimators are far less forgiving of that than MLJLinearModels, so
every estimator here is wrapped as::

    TransformedTargetRegressor(make_pipeline(StandardScaler(), estimator),
                               transformer=StandardScaler())

i.e. features standardised to zero mean / unit variance and the target standardised the
same way then inverted on :meth:`predict`, so scores stay in dollars.  This is exactly
the ``Standardizer |> TransformedTargetModel(model, target=Standardizer)`` pipeline that
the Julia neural-network script already used -- generalised to all ten models.  It leaves
``LinearRegression`` and ``RandomForestRegressor`` numerically unchanged (both are
equivariant under such affine re-parameterisations) and it rescues ``SGDRegressor``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone as _sk_clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    Ridge,
    SGDRegressor,
)
from sklearn.metrics import root_mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR

RANDOM_STATE = 123


@dataclass(frozen=True)
class ModelSpec:
    """A regressor plus the provenance needed to talk about it in the notebook."""

    key: str
    label: str
    julia_model: str
    julia_pkg: str
    family: str
    estimator: TransformedTargetRegressor

    def clone(self) -> TransformedTargetRegressor:
        """An unfitted copy (MLJ: calling the model type, e.g. ``LassoRegressor()``)."""
        return _sk_clone(self.estimator)


def _wrap(estimator) -> TransformedTargetRegressor:
    """Standardise features *and* target around ``estimator`` (see module docstring).

    ``transformer=StandardScaler()`` has to be passed explicitly: left at its default the
    meta-estimator applies the *identity* to ``y``, which silently leaves the SVR, NuSVR,
    MLP and penalised-linear fits chasing a target whose mean is 1.2 million dollars. This
    is the direct analogue of MLJ's ``TransformedTargetModel(model, target=Standardizer)``.
    """
    return TransformedTargetRegressor(
        make_pipeline(StandardScaler(), estimator), transformer=StandardScaler()
    )


def model_collection() -> dict[str, ModelSpec]:
    """Return the ten regressors keyed by short name (replaces ``LinearMolesCollections``)."""
    specs = [
        ModelSpec(
            "Linear",
            "Ordinary least squares",
            "LinearRegressor",
            "MLJLinearModels",
            "linear",
            _wrap(LinearRegression()),
        ),
        ModelSpec(
            "Huber",
            "Robust (Huber) regression",
            "RobustRegressor",
            "MLJLinearModels",
            "robust linear",
            _wrap(HuberRegressor(epsilon=1.35)),
        ),
        ModelSpec(
            "Ridge",
            "Ridge (L2) regression",
            "RidgeRegressor",
            "MLJLinearModels",
            "penalised linear",
            _wrap(Ridge()),
        ),
        ModelSpec(
            "Lasso",
            "Lasso (L1) regression",
            "LassoRegressor",
            "MLJLinearModels",
            "penalised linear",
            _wrap(Lasso(alpha=0.01, max_iter=10_000)),
        ),
        ModelSpec(
            "ElasticNet",
            "Elastic Net (L1 + L2)",
            "ElasticNetRegressor",
            "MLJLinearModels",
            "penalised linear",
            _wrap(ElasticNet(alpha=0.01, max_iter=10_000)),
        ),
        ModelSpec(
            "SGD",
            "Linear regression fitted by SGD",
            "SGDRegressor",
            "MLJScikitLearnInterface",
            "iterative linear",
            _wrap(SGDRegressor(random_state=RANDOM_STATE, max_iter=2000, tol=1e-4)),
        ),
        ModelSpec(
            "SVR",
            r"$\epsilon$-SVR, polynomial kernel",
            "EpsilonSVR",
            "LIBSVM",
            "kernel",
            # LIBSVM's default polynomial kernel has degree 3, as in the Julia script.
            _wrap(SVR(kernel="poly", degree=3, coef0=1.0, C=1.0)),
        ),
        ModelSpec(
            "NuSVR",
            r"$\nu$-SVR, RBF kernel",
            "NuSVR",
            "LIBSVM",
            "kernel",
            _wrap(NuSVR(nu=0.5, C=1.0)),
        ),
        ModelSpec(
            "MLP",
            "Feed-forward neural network",
            "NeuralNetworkRegressor",
            "MLJFlux",
            "neural",
            # Flux builder was Chain(Dense(n_in,64,relu), Dense(64,32,relu), Dense(32,n_out))
            _wrap(
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    max_iter=500,
                    early_stopping=True,
                    n_iter_no_change=16,
                    random_state=RANDOM_STATE,
                )
            ),
        ),
        ModelSpec(
            "RandomForest",
            "Random forest",
            "RandomForestRegressor",
            "DecisionTree",
            "tree ensemble",
            _wrap(
                RandomForestRegressor(
                    n_estimators=200,
                    min_samples_leaf=5,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            ),
        ),
    ]
    return {s.key: s for s in specs}


MODEL_KEYS: tuple[str, ...] = tuple(model_collection())


def rmsd(y_pred, y_true) -> float:
    """Julia ``StatsBase.rmsd`` == scikit-learn ``root_mean_squared_error``."""
    return float(root_mean_squared_error(y_true, y_pred))


def _final_estimator(machine: TransformedTargetRegressor):
    """Unwrap the fitted ``StandardScaler`` pipeline to reach the estimator itself."""
    pipe: Pipeline = machine.regressor_
    return pipe.steps[-1][1]


@dataclass
class FittedModel:
    """One fitted model plus everything the notebook wants to plot about it."""

    spec: ModelSpec
    machine: TransformedTargetRegressor
    y_pred: np.ndarray
    y_true: np.ndarray
    rmsd_test: float
    feature_names: list[str] = field(default_factory=list)
    rmsd_train: float | None = None
    fit_seconds: float | None = None

    @property
    def key(self) -> str:
        return self.spec.key

    @property
    def residuals(self) -> np.ndarray:
        """``yhat .- ytest`` from the Julia residue plots."""
        return np.asarray(self.y_pred, dtype=float) - np.asarray(self.y_true, dtype=float)

    @property
    def params(self) -> dict:
        """scikit-learn equivalent of MLJ ``fitted_params(mach)``."""
        est = _final_estimator(self.machine)
        return {
            k: v
            for k, v in est.get_params().items()
            if k not in {"random_state"} and not callable(v)
        }

    @property
    def feature_means(self) -> np.ndarray:
        """Raw-unit means of the training features (the ``Standardizer`` means)."""
        scaler = self.machine.regressor_.named_steps["standardscaler"]
        return np.asarray(scaler.mean_, dtype=float)

    @property
    def prediction_at_mean(self) -> float:
        """Fitted price when every feature sits at its training mean.

        For the linear models this *is* the intercept in dollar units; for the kernel,
        neural and forest models it is the closest useful analogue.
        """
        return float(self.machine.predict(np.array([self.feature_means]))[0])

    def coefficients(self) -> pd.Series:
        """Per-feature slopes in dollar units, i.e. d(price)/d(raw feature).

        scikit-learn reports linear coefficients in the standardised space created by the
        wrapping pipeline, so they are rescaled here (multiply by the target's sd, divide
        by each feature's sd).  The kernel, neural and forest models have no global slope,
        so the local slope at the training mean comes from central differences; trees are
        piecewise constant and can legitimately report 0.
        """
        est = _final_estimator(self.machine)
        scaler = self.machine.regressor_.named_steps["standardscaler"]
        x_scale = np.asarray(scaler.scale_, dtype=float)
        # scale_ of the target StandardScaler held in machine.transformer_ (1.0 if identity)
        y_scale = float(np.ravel(getattr(self.machine.transformer_, "scale_", 1.0))[0])
        coef = getattr(est, "coef_", None)
        if coef is not None and np.ndim(coef) <= 1:
            slope = np.asarray(coef, dtype=float).ravel() * y_scale / x_scale
        else:  # non-linear: central differences of the fitted predictor
            mean = np.asarray(scaler.mean_, dtype=float)
            h = x_scale * 1e-2
            step = []
            for j in range(len(self.feature_names)):
                plus, minus = mean.copy(), mean.copy()
                plus[j] += h[j]
                minus[j] -= h[j]
                yp = self.machine.predict(np.vstack([plus, minus]))
                step.append((float(yp[0]) - float(yp[1])) / (2 * h[j]))
            slope = np.asarray(step, dtype=float)
        return pd.Series(slope, index=self.feature_names, name=self.spec.key)


def fit_one(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    measure_train: bool = True,
) -> FittedModel:
    """``machine(spec.model, X, y) |> fit!`` followed by ``predict`` and ``rmsd``."""
    machine = spec.clone()
    t0 = time.perf_counter()
    machine.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    y_pred = np.asarray(machine.predict(X_test))
    out = FittedModel(
        spec=spec,
        machine=machine,
        y_pred=y_pred,
        y_true=np.asarray(y_test, dtype=float),
        rmsd_test=rmsd(y_pred, y_test),
        feature_names=list(X_train.columns),
        fit_seconds=elapsed,
    )
    if measure_train:
        out.rmsd_train = rmsd(np.asarray(machine.predict(X_train)), y_train)
    return out


def score_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    keys: Iterable[str] | None = None,
    *,
    specs: dict[str, ModelSpec] | None = None,
    verbose: bool = True,
) -> list[FittedModel]:
    """Fit and score several regressors -- the Julia list-comprehension, in full."""
    specs = specs or model_collection()
    chosen = [specs[k] for k in (keys or specs)]
    fitted: list[FittedModel] = []
    for spec in chosen:
        try:
            res = fit_one(spec, X_train, y_train, X_test, y_test)
        except Exception as exc:  # never lose the whole table to one bad fit
            print(f"{spec.key:>14}  FAILED: {type(exc).__name__}: {exc}")
            continue
        fitted.append(res)
        if verbose:
            print(f"{spec.key:>14}  RMSD = {res.rmsd_test:,.1f}")
    return fitted


def comparison_table(fitted: Iterable[FittedModel]) -> pd.DataFrame:
    """Rank the models by test RMSD -- the Julia ``for (k,v) in zip(...)`` loop as a table."""
    rows = [
        {
            "model": f.key,
            "description": f.spec.label,
            "julia model": f.spec.julia_model,
            "julia pkg": f.spec.julia_pkg,
            "family": f.spec.family,
            "RMSD (test)": f.rmsd_test,
            "RMSD (train)": f.rmsd_train,
            "fit (s)": f.fit_seconds,
        }
        for f in fitted
    ]
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values("RMSD (test)").reset_index(drop=True)


def coefficient_table(fitted: Iterable[FittedModel]) -> pd.DataFrame:
    """Local slopes of every fitted model (dollar units), one column per model."""
    if not isinstance(fitted, list):
        fitted = list(fitted)
    if not fitted:
        return pd.DataFrame()
    return pd.DataFrame({f.key: f.coefficients() for f in fitted})


if __name__ == "__main__":  # smoke test: python -m reg10.models
    from .data_processing import TARGET, get_data, partition, unpack

    df = get_data()
    X, y = unpack(df, TARGET)
    X_train, X_test, y_train, y_test = partition(X, y)
    print(f"{X_train.shape[0]} train rows, {X_test.shape[0]} test rows")
    fitted = score_models(X_train, y_train, X_test, y_test)
    print(comparison_table(fitted).to_string(index=False))
