from typing import List, Literal, Union

import numba as nb
import numpy as np

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


__ALL__ = [
    "indices_along_diagonal",
    "make_intercept",
    "make_model_array",
    "get_J_from_equation",
    "information_criteria_likelihood",
    "get_max_lambda",
    "get_adr_regularization_distance",
    "get_low_rank_regularization_distance",
]


def indices_along_diagonal(D: int) -> List:
    """This functions generates a list of indices that will go along
    the diagonal first and then go along the off-diagonals for a upper/lower
    triangular matrix.

    Args:
        D (int): Dimension of the matrix.

    Returns:
        List: List of indices
    """
    K = []
    for i in range(D):
        K.append(i)
        for j in range(D, i + 1, -1):
            K.append((K[-1] + j))
    return K


@nb.jit([
        "float32[:, :](float32[:], float32[:], float32[:, :], boolean)", 
        "float64[:, :](float64[:], float64[:], float64[:, :], boolean)",
    ], parallel=False)
def fast_vectorized_interpolate(x, xp, fp, ascending=True):
    dim = fp.shape[1]
    out = np.zeros((x.shape[0], dim), dtype=fp.dtype)
    if not ascending:
        x_asc = x[::-1]
        xp_asc = xp[::-1]

    for i in range(dim):
        if ascending:
            out[:, i] = np.interp(
                x=x,
                xp=xp,
                fp=fp[:, i],
            )
        else:
            out[:, i] = np.interp(
                x=x_asc,
                xp=xp_asc,
                fp=fp[:, i][::-1],
            )[::-1]
    return out


def make_intercept(n_observations: int) -> np.ndarray:
    """Make the intercept series as N x 1 array.

    Args:
        y (np.ndarray): Response variable $Y$

    Returns:
        np.ndarray: Intercept array.
    """
    return np.ones((n_observations, 1))


def make_model_array(X, eq, fit_intercept):
    n = X.shape[0]

    # TODO: Check difference between np.array and list more explicitly?
    if isinstance(eq, str) and (eq == "intercept"):
        if not fit_intercept:
            raise ValueError(
                "fit_intercept[param] is false, but equation says intercept."
            )
        out = make_intercept(n_observations=n)
    else:
        if isinstance(eq, str) and (eq == "all"):
            if isinstance(X, np.ndarray):
                out = X
            if HAS_PANDAS and isinstance(X, pd.DataFrame):
                out = X.to_numpy()
            if HAS_POLARS and isinstance(X, pl.DataFrame):
                out = X.to_numpy()
        elif isinstance(eq, np.ndarray) | isinstance(eq, list):
            if isinstance(X, np.ndarray):
                out = X[:, eq]
            if HAS_PANDAS and isinstance(X, pd.DataFrame):
                out = X.loc[:, eq]
            if HAS_POLARS and isinstance(X, pl.DataFrame):
                out = X.select(eq).to_numpy()
        else:
            raise ValueError("Did not understand equation. Please check.")

        if fit_intercept:
            out = np.hstack((make_intercept(n), out))

    return out


def get_J_from_equation(self, X: np.ndarray):
    J = {}
    for p in range(self.distribution.n_params):
        if isinstance(self.equation[p], str):
            if self.equation[p] == "all":
                J[p] = X.shape[1] + int(self.fit_intercept[p])
            if self.equation[p] == "intercept":
                J[p] = 1
        elif isinstance(self.equation[p], np.ndarray) or isinstance(
            self.equation[p], list
        ):
            J[p] = len(self.equation[p]) + int(self.fit_intercept[p])
        else:
            raise ValueError("Something unexpected happened")
    return J


def information_criteria_likelihood(
    log_likelihood: Union[float, np.ndarray],
    n_observations: Union[float, np.ndarray],
    n_parameters: Union[float, np.ndarray],
    ic: Literal["aic", "aicc", "bic", "hqc"],
) -> np.ndarray:
    if ic == "aic":
        value = 2 * n_parameters - 2 * log_likelihood
    elif ic == "aicc":
        value = (
            2 * n_parameters
            - 2 * log_likelihood
            + (
                (2 * n_parameters**2 + 2 * n_parameters)
                / (n_observations - n_parameters - 1)
            )
        )
    elif ic == "bic":
        value = n_parameters * np.log(n_observations) - 2 * log_likelihood
    elif ic == "hqc":
        value = n_parameters * np.log(np.log(n_observations)) - 2 * log_likelihood
    else:
        raise ValueError("Did not recognize ic.")
    return value


@nb.njit()
def get_max_lambda(x_gram: np.ndarray, y_gram: np.ndarray, is_regularized: np.ndarray):
    if np.all(is_regularized):
        max_lambda = np.max(np.abs(y_gram))
    elif np.sum(~is_regularized) == 1:
        intercept = y_gram[is_regularized] / np.diag(x_gram)[~is_regularized]
        max_lambda = np.max(
            np.abs(y_gram.flatten() - x_gram[~is_regularized] * intercept)
        )
    else:
        raise NotImplementedError("Currently not implemented")
    return max_lambda


def get_adr_regularization_distance(d: int, lower_diag: bool = True):
    if lower_diag:
        j, i = np.triu_indices(d, k=0)
    else:
        i, j = np.triu_indices(d, k=0)
    distance = np.abs(i - j)
    return distance


def get_low_rank_regularization_distance(d, r):
    return np.concatenate([np.repeat(i + 1, d) for i in range(r)])
