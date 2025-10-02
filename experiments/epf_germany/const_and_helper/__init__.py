import numpy as np
import pandas as pd

PLT_SAVE_OPTIONS = {
    "dpi": 300,
    "bbox_inches": "tight",
}
PLT_TEX_OPTIONS = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath,amssymb,amsfonts}",
}

# Constants
N_SIMS = 2500
FORGET = 0.0
H = 24
RANDOM_STATE = 123

FOLDER = "experiments/epf_germany/"
FOLDER_RESULTS = "experiments/epf_germany/results_revision/"
FOLDER_DATA = "experiments/epf_germany/"
FOLDER_FIGURES = "experiments/epf_germany/figures/"
FOLDER_TABLES = "experiments/epf_germany/tables/"


MODEL_NAMES_MAPPING = {
    "ar_ind": "LARX+N(0, $\sigma$)",
    "ar_dep": "LARX+N(0, $\Sigma$)",
    "ar_cp": "LARX+Adaptive CP",
    "ar_garch": "LARX+GARCH(1,1)",
    "no_copula": "oDistReg",
    "gaussian_copula": "oDistReg+GC",
    "sparse_gaussian_copula": "oDistReg+spGC",
    "cd_ols_ind": "oMvDistReg(t, CD, OLS, ind)",
    "cd_ols_dep": "oMvDistReg(t, CD, OLS)",
    "cd_lasso_dep": "oMvDistReg(t, CD, LASSO)",
    "mcd_ols_ind": "oMvDistReg(t, MCD, OLS, ind)",
    "mcd_ols_dep": "oMvDistReg(t, MCD, OLS)",
    "mcd_lasso_dep": "oMvDistReg(t, MCD, LASSO)",
    "lr_ols_ind": "oMvDistReg(t, LRA, OLS, ind)",
    "lr_ols_dep": "oMvDistReg(t, LRA, OLS)",
    "lr_lasso_dep": "oMvDistReg(t, LRA, LASSO)",
}


def boxcox(y: np.ndarray, power: float = 0.5, shift: float = 1.0):
    return np.sign(y) * (((np.abs(y) + shift) ** power - shift) / power)


def inv_boxcox(z: np.ndarray, power: float = 0.5, shift: float = 1.0):
    return np.sign(z) * ((np.abs(z) * power + shift) ** (1 / power) - shift)


# Functions for getting the models
def get_hourly_mean_model_data_index(
    X: pd.DataFrame,
    hour: int,
    linear: bool = True,
) -> np.ndarray:
    cols = X.columns
    # Lagged prices and average prices of the last day
    mask = cols.str.contains(f"h{hour}_price|price_l1", regex=True)
    mask = mask & (cols != "day_price_l1_avg")
    mask = mask & ~cols.str.contains("roll")

    if linear:
        # Resdiual Load & Fuel prices
        mask = mask | cols.str.contains(f"h{hour}_residload_linear")
        # mask = mask | cols.str.contains("day_residload_linear")
        mask = mask | (
            cols.str.contains("gas|oil|eua|coal") & ~cols.str.contains("basis")
        )
        mask = mask & ~cols.str.contains("margcost")

    else:
        # Spline of residual load and fuel prices
        mask = mask | (
            cols.str.contains(f"h{hour}_")
            & cols.str.contains("basis")
            & cols.str.contains("residload|gas|coal|oil")
        )

    # Dummies
    mask = mask | cols.str.contains("binary")
    return mask


def get_scale_data(X, hour: int) -> np.ndarray:
    cols = X.columns
    mask = cols.str.contains("binary")
    mask = mask | cols.str.contains(f"gas|coal|eua|oil|h{hour}_residload_linear")
    mask = mask & ~cols.str.contains("basis|margcost")
    mask = mask | cols.str.contains(f"h{hour}_price_roll7_std")
    return mask


def get_cholesky_data(X, h0, h1):
    cols = X.columns

    mask = cols.str.contains("binary")
    mask = mask | cols.str.contains(f"gas|coal|eua|oil|h{h0}_residload_linear")
    mask = mask & ~cols.str.contains("basis|margcost")

    if h0 == h1:
        mask = mask | X.columns.str.contains(f"h{h0}_price_roll7_invstd")
    else:
        mask = mask | X.columns.str.contains(f"price_h{h0}_h{h1}_costd")
    return mask


def get_daily_data_index(X: pd.DataFrame) -> np.ndarray:
    cols = X.columns
    mask = cols.str.contains("binary")
    mask = mask | cols.str.contains("day_residload|gas|coal|eua|oil")
    mask = mask & ~cols.str.contains("basis|margcost")
    return mask


def get_low_rank_data_index(X, hour, r):
    cols = X.columns
    if r == 0:
        mask = cols.str.contains(f"gas|coal|eua|oil|h{hour}_residload_linear")
        mask = mask & ~cols.str.contains("basis|margcost")
        mask = mask | X.columns.str.contains(f"h{hour}_price_roll7_invstd")
    if r == 1:
        mask = cols.str.contains("binary")
    return mask


def get_to_scale(X: pd.DataFrame) -> np.ndarray:
    index = ~X.columns.str.contains("binary")
    index = index & ~(X.columns == "constant")
    return index
