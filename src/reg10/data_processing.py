"""Data loading and preparation.

Port of ``data-processing.jl`` (Julia / MLJ / DataFrames) to pandas.

Original workflow::

    df = CSV.File("./DataSource/usa_housing.csv") |> DataFrame |> dropmissing
    coerce(df, Count => Continuous)                       # integer -> float
    rename!(df, [1 => :AreaIncome, 2 => :HouseAge, 3 => :HouseRooms,
                 4 => :HouseBedromms, 5 => :AreaPopulation])
    select!(df, 1:6)                                      # drop the free-text Address

Data source: <https://www.kaggle.com/datasets/ajayp1/usa-housing> (as used by the
Kaggle notebook *Practical Introduction to 10 Regression Algorithms*).
"""

from __future__ import annotations

import functools
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

#: Repo layout — ``data/usa_housing.csv`` sits next to the ``src/`` package.
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CSV_NAME = "usa_housing"

#: Positional renames applied to the first five columns (column 6 is ``Price``).
#: Note: the Julia original used the misspelled ``:HouseBedromms``; corrected here.
COLUMN_MAP = {
    "Avg. Area Income": "AreaIncome",
    "Avg. Area House Age": "HouseAge",
    "Avg. Area Number of Rooms": "HouseRooms",
    "Avg. Area Number of Bedrooms": "HouseBedrooms",
    "Area Population": "AreaPopulation",
    "Price": "Price",
}

FEATURES = ["AreaIncome", "HouseAge", "HouseRooms", "HouseBedrooms", "AreaPopulation"]
TARGET = "Price"

#: Seed used by the Julia code (``unpack(..., rng=123)``); kept for comparability.
RANDOM_STATE = 123
#: Julia ``partition((X, y), 0.7, ...)`` -> 70% train / 30% test.
TRAIN_SIZE = 0.7


def load_data(csv_dir: Path | str = DATA_DIR, name: str = CSV_NAME) -> pd.DataFrame:
    """Read the CSV, coerce numerics, drop missing rows, rename and select columns.

    Equivalent of the Julia ``data_prepare(str)`` + ``rename!`` + ``select!`` chain.
    """
    raw = pd.read_csv(Path(csv_dir) / f"{name}.csv")

    # rename!(df, [1 => :AreaIncome, ...])
    df = raw.rename(columns=COLUMN_MAP)

    # coerce(Count => Continuous): every modelling column is numeric; the free-text
    # Address column is not, so it is dropped here rather than by a select!(1:6).
    columns = [c for c in df.columns if c in FEATURES + [TARGET]]
    if len(columns) != 6:  # positional fallback, like the Julia code
        columns = list(df.columns[:6])
    df = df[columns].apply(pd.to_numeric, errors="coerce").astype("float64")

    # dropmissing
    return df.dropna().reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def get_data() -> pd.DataFrame:
    """Cached copy of the tidy modelling frame (mirrors the Julia ``get_data()``)."""
    return load_data()


def unpack(df: pd.DataFrame, target: str = TARGET) -> tuple[pd.DataFrame, pd.Series]:
    """``y, X = unpack(df, ==(:Price))`` -> ``(X, y)``."""
    return df.drop(columns=[target]), df[target]


def partition(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = TRAIN_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Julia ``partition((X, y), 0.7, multi=true)`` -> ``(Xtr, Xte, ytr, yte)``."""
    return train_test_split(X, y, train_size=train_size, random_state=random_state)


def every_third(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Port of the exploratory ``xtest = df[1:3:end, 1:5]; ytest = df[1:3:end, 6]``.

    .. warning::
       Overlaps the training rows, so scores computed against it are optimistic.
       Retained only for faithful reproduction of the Julia notebooks; prefer
       :func:`partition` for anything you intend to report.
    """
    sub = df.iloc[::3]
    return sub[FEATURES], sub[TARGET]


if __name__ == "__main__":  # smoke test: python -m reg10.data_processing
    frame = get_data()
    print(frame.head())
    print(frame.describe().round(3))
    print("shape:", frame.shape, "| columns:", list(frame.columns))
