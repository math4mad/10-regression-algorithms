# 10 Regression Algorithms — Julia → Python

[![Publish Quarto note to gh-pages](https://github.com/math4mad/10-regression-algorithms/actions/workflows/publish.yml/badge.svg)](https://github.com/math4mad/10-regression-algorithms/actions/workflows/publish.yml)
&nbsp; Site: <https://math4mad.github.io/10-regression-algorithms/>

A refactor of my earlier Julia project
[*Practical Introduction to 10 Regression Algorithm*](https://github.com/math4mad/Julia-ML/tree/main/Practical%20Introduction%20to%2010%20Regression%20Algorithm)
(MLJ.jl + GLMakie) into Python (pandas + scikit-learn + matplotlib), with the explanatory
notebook rewritten as a Quarto document that executes Python.

Everything fits the four libraries you already know: **numpy**, **pandas**, **matplotlib**,
**scikit-learn**.

## Branches

| branch    | contents                                                              |
|-----------|-----------------------------------------------------------------------|
| `main`    | snapshot of the original **Julia** source (`*.jl` + the Julia note)     |
| `python`  | the **Python** refactor: package, scripts, Quarto note, CI              |

`python` is where the work happens; the Julia sources were deleted there once the port was
verified, so `git diff main python` is the whole refactor.

## The ten algorithms

Data: `data/usa_housing.csv` — 5,000 rows of area-level US housing statistics
(Kaggle *USA Real Estate*, as used by
[faressayah's notebook](https://www.kaggle.com/code/faressayah/practical-introduction-to-10-regression-algorithm)).
Five predictors (`AreaIncome`, `HouseAge`, `HouseRooms`, `HouseBedrooms`, `AreaPopulation`)
predict `Price`; 70/30 split with `random_state=123`; scored with RMSD
(= `root_mean_squared_error`).

| #  | scikit-learn                          | Julia (MLJ)                | Julia package             |
|----|---------------------------------------|----------------------------|---------------------------|
| 1  | `LinearRegression`                    | `LinearRegressor`          | MLJLinearModels           |
| 2  | `HuberRegressor`                      | `RobustRegressor`          | MLJLinearModels           |
| 3  | `Ridge`                               | `RidgeRegressor`           | MLJLinearModels           |
| 4  | `Lasso`                               | `LassoRegressor`           | MLJLinearModels           |
| 5  | `ElasticNet`                          | `ElasticNetRegressor`      | MLJLinearModels           |
| 6  | `SGDRegressor`                        | `SGDRegressor`             | MLJScikitLearnInterface   |
| 7  | `SVR(kernel="poly")`                  | `EpsilonSVR`               | LIBSVM                    |
| 8  | `NuSVR`                               | `NuSVR`                    | LIBSVM                    |
| 9  | `MLPRegressor(hidden_layer_sizes=(64, 32))` | `NeuralNetworkRegressor` | MLJFlux / Flux      |
| 10 | `RandomForestRegressor`               | `RandomForestRegressor`    | DecisionTree              |

Each estimator is wrapped as
`TransformedTargetRegressor(make_pipeline(StandardScaler(), model), transformer=StandardScaler())`
so that features *and* the dollar-scale target are standardised before fitting — the
`Standardizer |> TransformedTargetModel(...)` pipeline the Julia neural-network script used,
applied uniformly. See *Porting notes* in the rendered notebook for the five places where
the Python version intentionally departs from the Julia original.

## Layout

```
_quarto.yml            Quarto website config (renders index.qmd into _site/)
index.qmd              the notebook: EDA → 10 models → residuals → porting notes
src/reg10/
  data_processing.py   port of data-processing.jl   (load, coerce, rename, split)
  models.py            port of 10-algorithm-compare.jl (registry + RMSD harness)
  plots.py             ports of the GLMakie figures (pairs, heatmap, stem, hist, …)
scripts/
  pair_plot.py         port of cor-plot.jl
  compare_algorithms.py port of 10-algorithm-compare.jl
  linear_regressor.py, ridge_regressor.py, lasso_regressor.py, robust_regressor.py,
  elastic_net_regressor.py, sgd_regressor.py, epsilon_svr.py, nu_svr.py,
  neural_network_regression.py, random_forest_regressor.py
data/usa_housing.csv   5,000 rows
.github/workflows/publish.yml  render + deploy _site/ to the gh-pages branch
Makefile               venv / render / smoke-test shortcuts
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m reg10.models            # fit all ten, print the RMSD table
python scripts/compare_algorithms.py
python scripts/lasso_regressor.py # one algorithm, mirrors lasso-regressor.jl
python scripts/pair_plot.py

make render                       # quarto render -> _site/
```

`reg10` is a plain source package under `src/` with no install step, so give the
interpreter a path to it (`export PYTHONPATH=src`) or run through `make`, which does that
for you. The scripts in `scripts/` add it themselves.

`make` exports `JUPYTER_PATH=$(PWD)/.venv/share/jupyter` so Quarto uses the venv's
`python3` kernel instead of a stale user-level kernelspec; if you call `quarto render`
directly, export it yourself.

## Publishing

Pushing to the `python` branch runs `.github/workflows/publish.yml`: it installs
`requirements.txt`, sets up Quarto, runs `quarto render index.qmd`, and pushes `_site/` to
the `gh-pages` branch, which GitHub serves at
<https://math4mad.github.io/10-regression-algorithms/>.

## Acknowledgements

* Original notebook structure: faressayah, *Practical Introduction to 10 Regression Algorithms* (Kaggle)
* MLJ model names and defaults: the Julia sources on the `main` branch of this repo
