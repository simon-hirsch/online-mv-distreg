import time

import numpy as np
import pandas as pd
import rolch
import scipy.stats as st
from const_and_helper import (
    N_SIMS,
    H,
    get_cholesky_data,
    get_daily_data_index,
    get_hourly_mean_model_data_index,
)
from tqdm import tqdm

print(rolch.__version__)

df_X = pd.read_csv("prepared_x.csv", index_col=0)
df_y = pd.read_csv("prepared_y.csv", index_col=0)
X_numpy = df_X.drop(["flag"], axis=1).values.astype(float)
y_numpy = df_y.values.astype(float)


# Define indices
IDX_TRAIN = df_X.flag.to_numpy() == "train"
IDX_TEST = df_X.flag.to_numpy() == "test"

N_TRAIN = np.sum(IDX_TRAIN)
N_TEST = np.sum(IDX_TEST)
N = df_X.shape[0]


# Fit, predict once to run jit-compilation
X = df_X.drop(["flag"], axis=1)
y = df_y
h = 0
n = N_TRAIN
IDX_DATA_H = get_hourly_mean_model_data_index(X, hour=h)

model = rolch.OnlineLasso(
    fit_intercept=True,
    regularize_intercept=False,
    ic="bic",
    scale_inputs=True,
)
model.fit(
    X=X_numpy[:n, IDX_DATA_H],
    y=y_numpy[:n, h],
)
model.predict(
    X=np.expand_dims(X_numpy[n, IDX_DATA_H], 0),
)
model.update(
    X=np.expand_dims(X_numpy[n, IDX_DATA_H], 0),
    y=np.expand_dims(y_numpy[n, h], 0),
)

# Run univariate benchmark models
# Gaussian fixed covariance structure
models = {}
predictions_loc = np.zeros((N_TEST, H))
predictions_cov = np.zeros((N_TEST, H, H))
residuals = np.zeros((N - 1, H))
timings = np.zeros((N_TEST, H + 1))

for i, n in tqdm(enumerate(range(N_TRAIN, N))):
    for h in range(H):
        IDX_DATA_H = get_hourly_mean_model_data_index(X, hour=h)
        if i == 0:
            models[h] = rolch.OnlineLinearModel(
                method="lasso",
                fit_intercept=True,
                regularize_intercept=False,
                ic="bic",
                scale_inputs=True,
                forget=0.0,
            )
            start = time.time()
            models[h].fit(
                X=X_numpy[:n, IDX_DATA_H],
                y=y_numpy[:n, h],
            )

            residuals[:n, h] = y_numpy[:n, h] - models[h].predict(
                X=X_numpy[:n, IDX_DATA_H],
            )

            stop = time.time()
        else:
            start = time.time()
            models[h].update(
                X=np.expand_dims(X_numpy[n - 1, IDX_DATA_H], 0),
                y=np.expand_dims(y_numpy[n - 1, h], 0),
            )
            residuals[n - 1, h] = (
                y_numpy[n - 1, h]
                - models[h]
                .predict(X=np.expand_dims(X_numpy[n - 1, IDX_DATA_H], 0))
                .item()
            )
            stop = time.time()

        timings[i, h] = stop - start
        predictions_loc[i, h] = (
            models[h].predict(X=np.expand_dims(X_numpy[n, IDX_DATA_H], 0)).item()
        )
    if i == 0:
        start = time.time()
        predictions_cov[i, :, :] = np.cov(residuals[np.arange(n), :], rowvar=False)
        stop = time.time()
    else:
        start = time.time()
        predictions_cov[i, ...] = (1 / n) * (
            (n - 1) * predictions_cov[i - 1, :, :]
            + np.outer(residuals[[n - 1], :], residuals[[n - 1], :])
        )
        stop = time.time()
    timings[i, H] = stop - start

# We have two models because we assume 1 with independence
RANDOM_STATE = 123
N_MODELS_AR = 2

timings = np.tile(timings.sum(axis=1), (N_MODELS_AR, 1)).T
simulations_ar = np.zeros((N_TEST, N_MODELS_AR, N_SIMS, H))
for t in range(N_TEST):
    ## Multivariate Normal
    simulations_ar[t, 0, ...] = st.multivariate_normal(
        mean=predictions_loc[t],
        cov=np.diag(predictions_cov[t].diagonal()),
    ).rvs((N_SIMS,), random_state=RANDOM_STATE + t)

    # Multivariate Normal
    simulations_ar[t, 1, ...] = st.multivariate_normal(
        mean=predictions_loc[t],
        cov=predictions_cov[t],
    ).rvs((N_SIMS, 1), random_state=RANDOM_STATE + t)


# Save predictions
np.savez(
    file="results/pred_univariate_benchmark.npz",
    predictions_loc=predictions_loc,
    predictions_cov=predictions_cov,
    residuals=residuals,
    timings=timings,
)

np.savez(
    file="results/sims_univariate_benchmark.npz",
    simulations=simulations_ar,
)


# Make univariate model predictions for the copula
# Copula will be run in separate file
equations = {
    h: {
        0: np.arange(X.shape[1])[get_hourly_mean_model_data_index(X, h)],
        1: np.arange(X.shape[1])[
            get_cholesky_data(X, h, h)
        ],  # same model as on the diagonal
        2: np.arange(X.shape[1])[get_daily_data_index(X)],
    }
    for h in range(H)
}

models_gamlss = {}
insample_probabilistic = np.zeros((N - 1, H, 3))
predictions_probabilistic = np.zeros((N_TEST, H, 3))
timings = np.zeros((N_TEST, H))

for i, n in tqdm(enumerate(range(N_TRAIN, N))):
    for h in range(H):
        if i == 0:
            models_gamlss[h] = rolch.OnlineGamlss(
                distribution=rolch.DistributionT(),
                equation=equations[h],
                method="lasso",
                fit_intercept=True,
                regularize_intercept=False,
                ic="bic",
                scale_inputs=True,
                forget=0.0,
                debug=False,
                verbose=0,
            )
            start = time.time()
            models_gamlss[h].fit(
                X=X_numpy[:n, :],
                y=y_numpy[:n, h],
            )
            stop = time.time()
        else:
            start = time.time()
            models_gamlss[h].update(
                X=np.expand_dims(X_numpy[n - 1, :], 0),
                y=np.expand_dims(y_numpy[n - 1, h], 0),
            )
            stop = time.time()

        timings[i, h] = stop - start

        if i == 0:
            insample_probabilistic[:n, h, :] = models_gamlss[h].predict(
                X=X_numpy[:n, :],
            )
        else:
            insample_probabilistic[n - 1, h, :] = models_gamlss[h].predict(
                X=np.expand_dims(X_numpy[n - 1, :], 0),
            )

        predictions_probabilistic[i, h, :] = models_gamlss[h].predict(
            X=np.expand_dims(X_numpy[n, :], 0),
        )


np.savez_compressed(
    file="results/univariate_predictions_distreg.npz",
    predictions_outofsample=predictions_probabilistic,
    predictions_insample=insample_probabilistic,
    timings=timings,
)
