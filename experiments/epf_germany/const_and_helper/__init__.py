import numpy as np
import pandas as pd

# Constants
N_SIMS = 2500
H = 24


# Functions for getting the models
def get_hourly_mean_model_data_index(
    X: pd.DataFrame,
    hour: int,
) -> np.ndarray:
    cols = X.columns
    index = (
        cols.str.contains(f"h{hour}_")
        | cols.str.contains("l1")
        | cols.str.contains("binary")
        | cols.str.contains("day")
    ) & (
        cols != "day_p_l1_avg"  # Avoid multi-colinearity with lags L1 data
    )
    index = index & ~cols.str.contains("basis")
    index = index & ~cols.str.contains("roll")
    index = index & ~cols.str.contains("costd")
    return index


def get_cholesky_data(X, h0, h1):
    index = X.columns.str.contains("day")
    index = index & ~X.columns.str.contains("min|max|q10|q90")
    index = index | X.columns.str.contains(f"h{h0}_res")
    index = index | X.columns.str.contains(f"h{h0}_load")

    if h0 == h1:
        index = index | X.columns.str.contains(f"h{h0}_price_roll7_std")
    else:
        index = index | X.columns.str.contains(f"price_h{h0}_h{h1}_costd")
    return index


def get_daily_data_index(X: pd.DataFrame) -> np.ndarray:
    index = X.columns.str.contains("binary") | X.columns.str.contains("day")
    index = index & ~X.columns.str.contains("min|max|q10|q90")
    return index


def get_low_rank_data_index(X, hour, r):
    cols = X.columns
    if r == 0:
        index = (
            cols.str.contains(f"h{hour}_res")
            | cols.str.contains(f"h{hour}_load")
            | cols.str.contains("day")
        )
        index & ~X.columns.str.contains("min|max|q10|q90")
        index = index | X.columns.str.contains(f"h{hour}_price_roll7_std")
        index = index & ~cols.str.contains("basis")
    if r == 1:
        index = cols.str.contains("binary")
    return index


def get_to_scale(X: pd.DataFrame) -> np.ndarray:
    index = ~X.columns.str.contains("binary")
    index = index & ~(X.columns == "constant")
    return index
