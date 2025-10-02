# %%
# Allow running as notebook and as script
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# %%
import time

import numpy as np
import pandas as pd
import scipy.stats as st
from const_and_helper import (
    FOLDER_DATA,
    FOLDER_RESULTS,
    FORGET,
    N_SIMS,
    RANDOM_STATE,
    H,
    get_cholesky_data,
    get_daily_data_index,
    get_hourly_mean_model_data_index,
)
from ondil.distributions import StudentT
from ondil.estimators import OnlineDistributionalRegression, OnlineLinearModel
from ondil.links import (  # noqa: F401
    Identity,
    InverseSoftPlusShiftTwo,
    InverseSoftPlusShiftValue,
    Log,
    LogShiftTwo,
    Sqrt,
)
from tqdm import tqdm

# %%
df_X = pd.read_csv(os.path.join(FOLDER_DATA, "prepared_long_x.csv"), index_col=0)
df_y = pd.read_csv(os.path.join(FOLDER_DATA, "prepared_long_y.csv"), index_col=0)

X = df_X.drop(["flag"], axis=1)
X_numpy = X.values.astype(np.float32)
y_numpy = df_y.values.astype(np.float32)

# Define indices
IDX_TRAIN = df_X.flag.to_numpy() == "train"
IDX_TEST = df_X.flag.to_numpy() == "test"

N_TRAIN = np.sum(IDX_TRAIN)
N_TEST = np.sum(IDX_TEST)
N = df_X.shape[0]

# %%
# for scaling/transformation
# y is already transformed
scaling_m = pd.read_csv(os.path.join(FOLDER_DATA, "prices_means.csv"), index_col=0)
scaling_s = pd.read_csv(os.path.join(FOLDER_DATA, "prices_variances.csv"), index_col=0)

scaling_m_ins = scaling_m.values.astype(np.float32)[:-1, :]
scaling_s_ins = scaling_s.values.astype(np.float32)[:-1, :]
# We need to shift the scaling_m and scaling_s by one day
# Because we need the inverse transformation of the to happen with the
# correct information set!
scaling_m = scaling_m.values.astype(np.float32)[np.roll(IDX_TEST, -1), :]
scaling_s = scaling_s.values.astype(np.float32)[np.roll(IDX_TEST, -1), :]


prices = pd.read_csv(os.path.join(FOLDER_DATA, "prices_untransformed.csv"), index_col=0)
prices = prices.values.astype(np.float32)
prices_test = prices[IDX_TEST, :]

# %%
# Fit, predict once to run jit-compilation
IDX_DATA_H = get_hourly_mean_model_data_index(df_X.drop(["flag"], axis=1), hour=0)

model = OnlineLinearModel(
    method="lasso",
    fit_intercept=True,
    regularize_intercept=False,
    ic="bic",
    scale_inputs=True,
)
model.fit(
    X=X_numpy[:N_TRAIN, IDX_DATA_H],
    y=y_numpy[:N_TRAIN, 0],
)
model.predict(
    X=np.expand_dims(X_numpy[N_TRAIN, IDX_DATA_H], 0),
)
model.update(
    X=np.expand_dims(X_numpy[N_TRAIN, IDX_DATA_H], 0),
    y=np.expand_dims(y_numpy[N_TRAIN, 0], 0),
)

# %%

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
        TO_SCALE = X.loc[:, IDX_DATA_H].columns.str.contains(
            "load|gas|res|coal|eua|oil"
        )
        if i == 0:
            models[h] = OnlineLinearModel(
                method="lasso",
                fit_intercept=True,
                regularize_intercept=False,
                ic="bic",
                scale_inputs=TO_SCALE,
                forget=FORGET,
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

        predictions_loc[i, h] = (
            models[h]
            .predict(
                X=np.expand_dims(X_numpy[n, IDX_DATA_H], 0),
            )
            .item()
        )
        timings[i, h] = stop - start

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

# %%
# Run simuations
# We have two models because we assume 1 with independence
N_MODELS_AR = 2

# Re-scale the predictions to the original scale
location = np.zeros_like(predictions_loc)
scale = np.zeros_like(predictions_cov)

diag_s = np.zeros((N_TEST, H, H))
diag_s[:, range(H), range(H)] = scaling_s

location = predictions_loc * scaling_s + scaling_m
scale = diag_s @ predictions_cov @ diag_s.swapaxes(-1, -2)

# %%
# Create timings and simulations
timings = np.tile(timings.sum(axis=1), (N_MODELS_AR, 1)).T
simulations_ar = np.zeros((N_TEST, N_MODELS_AR, N_SIMS, H))
for t in range(N_TEST):
    ## Multivariate Normal
    simulations_ar[t, 0, ...] = st.multivariate_normal(
        mean=location[t],
        cov=np.diag(scale[t].diagonal()),
    ).rvs((N_SIMS, 1), random_state=RANDOM_STATE + t)

    # Multivariate Normal
    simulations_ar[t, 1, ...] = st.multivariate_normal(
        mean=location[t],
        cov=scale[t],
    ).rvs((N_SIMS, 1), random_state=RANDOM_STATE + t)

# %%
# Save predictions
MODEL_NAMES_AR = ["ar_ind", "ar_dep"]

np.savez(
    file=os.path.join(FOLDER_RESULTS, "pred_univariate_benchmark.npz"),
    model_names=MODEL_NAMES_AR,
    predictions_loc_scaled=predictions_loc,
    predictions_cov_scaled=predictions_cov,
    predictions_loc=location,
    predictions_cov=scale,
    residuals=residuals,
    timings=timings,
)
np.savez(
    file=os.path.join(FOLDER_RESULTS, "sims_univariate_benchmark.npz"),
    model_names=MODEL_NAMES_AR,
    simulations=simulations_ar,
)


# %%
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

distribution = StudentT(
    loc_link=Identity(),
    scale_link=Log(),
    tail_link=InverseSoftPlusShiftValue(2.1),
)

models_gamlss = {}
insample_probabilistic = np.zeros((N - 1, H, 3))
predictions_probabilistic = np.zeros((N_TEST, H, 3))
timings = np.zeros((N_TEST, H))

TO_SCALE_DISTREG = X.loc[:, :].columns.str.contains("load|gas|res|coal|eua")

for i, n in tqdm(enumerate(range(N_TRAIN, N))):
    for h in range(H):
        if i == 0:
            models_gamlss[h] = OnlineDistributionalRegression(
                distribution=distribution,
                equation=equations[h],
                method="lasso",
                fit_intercept=True,
                regularize_intercept=False,
                ic="bic",
                scale_inputs=TO_SCALE_DISTREG,
                forget=FORGET,
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
            insample_probabilistic[:n, h, :] = models_gamlss[
                h
            ].predict_distribution_parameters(
                X=X_numpy[:n, :],
            )
        else:
            insample_probabilistic[n - 1, h, :] = models_gamlss[
                h
            ].predict_distribution_parameters(
                X=np.expand_dims(X_numpy[n - 1, :], 0),
            )

        predictions_probabilistic[i, h, :] = models_gamlss[
            h
        ].predict_distribution_parameters(
            X=np.expand_dims(X_numpy[n, :], 0),
        )

# %%
# Rescale the predictions
predictions_rescaled = np.zeros_like(predictions_probabilistic)
insample_rescaled = np.zeros_like(insample_probabilistic)

predictions_rescaled[..., 0] = predictions_probabilistic[..., 0] * scaling_s + scaling_m
predictions_rescaled[..., 1] = predictions_probabilistic[..., 1] * scaling_s
predictions_rescaled[..., 2] = predictions_probabilistic[..., 2]

IDX_INSAMPLE = np.arange(insample_probabilistic.shape[0])

insample_rescaled[..., 0] = (
    insample_probabilistic[..., 0] * scaling_s_ins + scaling_m_ins
)
insample_rescaled[..., 1] = insample_probabilistic[..., 1] * scaling_s_ins
insample_rescaled[..., 2] = insample_probabilistic[..., 2]

# %%
np.savez_compressed(
    file=os.path.join(FOLDER_RESULTS, "univariate_predictions_distreg.npz"),
    predictions_outofsample_scaled=predictions_probabilistic,
    predictions_insample_scaled=insample_probabilistic,
    predictions_outofsample=predictions_rescaled,
    predictions_insample=insample_rescaled,
    timings=timings,
)
