import numba as nb
import numpy as np


@nb.jit()
def joint_prediction_band(simulations: np.ndarray, alpha: float, power: int = 2):
    pred = simulations.sum(axis=0) / simulations.shape[0]
    sims = simulations.copy()

    iterations = int(np.ceil(simulations.shape[0] * alpha))
    for i in range(iterations):
        idx_min = sims.argmin(0)
        idx_max = sims.argmax(0)
        idx_extreme = np.unique(np.stack((idx_min, idx_max)))
        distance = np.sum(np.abs(sims[idx_extreme, :] - pred) ** power, 1)
        remove = idx_extreme[np.argmax(distance)]
        idx = np.array([i for i in range(sims.shape[0]) if i != remove])
        sims = sims[idx, :]

    band = np.zeros((2, sims.shape[1]))
    for i in range(sims.shape[1]):
        band[0, i] = min(sims[:, i])
        band[1, i] = max(sims[:, i])
    return band


# %%
# Define some scores
## DSS Currently not in the scoringrules package
@nb.guvectorize(
    [
        "void(float32[:], float32[:,:], float32[:])",
        "void(float64[:], float64[:,:], float64[:])",
    ],
    "(d),(m,d)->()",
)
def dawid_sebastiani_scoore(obs: np.ndarray, fct: np.ndarray, out: np.ndarray):
    M = fct.shape[0]
    bias = obs - (np.sum(fct, axis=0) / M)
    cov = np.cov(fct, rowvar=False).astype(bias.dtype)
    prec = np.linalg.inv(cov).astype(bias.dtype)
    log_det = np.log(np.linalg.det(cov))
    bias_precision = bias.T @ prec @ bias
    out[0] = log_det + bias_precision


# As implemented in the scoringrules package
# https://github.com/frazane/scoringrules/blob/main/scoringrules/core/energy/_gufuncs.py
# Current version of the package does not yet include this (as of 2024-09-14)
# Sam and Francesco need to build a new release
@nb.guvectorize(
    [
        "void(float32[:], float32[:,:], float32[:])",
        "void(float64[:], float64[:,:], float64[:])",
    ],
    "(d),(m,d)->()",
)
def energy_score_akr(obs: np.ndarray, fct: np.ndarray, out: np.ndarray):
    """Compute the Energy Score for a finite ensemble using the approximate kernel representation."""
    M = fct.shape[0]

    e_1 = 0.0
    e_2 = 0.0
    for i in range(M):
        e_1 += float(np.linalg.norm(fct[i] - obs))
        e_2 += float(np.linalg.norm(fct[i] - fct[i - 1]))

    out[0] = e_1 / M - 0.5 * 1 / M * e_2


@nb.guvectorize(
    [
        "void(float32[:], float32[:,:], float32[:])",
        "void(float64[:], float64[:,:], float64[:])",
    ],
    "(d),(m,d)->()",
)
def energy_score_fast(
    obs: np.ndarray,
    fct: np.ndarray,
    out: np.ndarray,
):
    """Compute the Energy Score for a finite ensemble."""
    M = fct.shape[0]

    e_1 = 0.0
    e_2 = 0.0
    for i in range(M):
        e_1 += float(np.linalg.norm(fct[i] - obs))
        for j in range(i + 1, M):
            e_2 += 2 * float(np.linalg.norm(fct[i] - fct[j]))

    out[0] = e_1 / M - 0.5 / (M**2) * e_2
