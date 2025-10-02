# %%
# Allow running as notebook and as script
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# %%
import time
from itertools import product

import numpy as np
import pandas as pd
import scoringrules as sr
from const_and_helper import (
    FOLDER_DATA,
    FOLDER_RESULTS,
    FORGET,
    N_SIMS,
    H,
    get_hourly_mean_model_data_index,
)
from const_and_helper.conformal import aci_clipped
from const_and_helper.distribution import (
    OnlineGaussianCopula,
    OnlineSparseGaussianCopula,
)
from ondil.estimators import OnlineLinearModel
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
scaling_m = scaling_m.values.astype(np.float32)  # [np.roll(IDX_TEST, -1), :]
scaling_s = scaling_s.values.astype(np.float32)  # [np.roll(IDX_TEST, -1), :]


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
CALIB_SET = 200

scores = np.zeros((CALIB_SET + N_TEST, H))
predictions = np.zeros((CALIB_SET + N_TEST, H))
timings = np.zeros((CALIB_SET + N_TEST, H + 1))
residuals = np.zeros((N, H))
predictions_in_sample = np.zeros((N - 1, H))

for i, n in tqdm(enumerate(range(N_TRAIN - CALIB_SET, N))):
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
            stop = time.time()
            predictions_in_sample[:n, h] = (
                models[h].predict(X=X_numpy[:n, IDX_DATA_H]) * scaling_s[:n, h]
                + scaling_m[:n, h]
            )
        else:
            start = time.time()
            models[h].update(
                X=np.expand_dims(X_numpy[n - 1, IDX_DATA_H], 0),
                y=np.expand_dims(y_numpy[n - 1, h], 0),
            )
            stop = time.time()
            predictions_in_sample[n - 1, h] = (
                models[h]
                .predict(X=np.expand_dims(X_numpy[n - 1, IDX_DATA_H], 0))
                .item()
                * scaling_s[n - 1, h]
                + scaling_m[n - 1, h]
            )
        # Collect scores for calibration
        pred = (
            models[h]
            .predict(
                X=np.expand_dims(X_numpy[n, IDX_DATA_H], 0),
            )
            .item()
        )
        predictions[i, h] = pred * scaling_s[n - 1, h] + scaling_m[n - 1, h]
        scores[i, h] = prices[n, h] - predictions[i, h]
        timings[i, h] = stop - start


# %%
# Calculate scores
# Residuals are already calculated
# We use the absolute value of the residuals as scores

scores = np.abs(scores)
ALPHA = 1 - np.linspace(0.01, 0.99, 99)
# 2, 4, 6, ..., 98 % prediction intervals
# implies the Q1/99, Q2/98, ..., Q49/51
Q = np.linspace(0.005, 0.995, 199)
QQ = np.concatenate(([0], Q, [1]))
CENTER = 100

# %%
in_sample_cpd = np.zeros((N - 1, H, len(Q) + 2))
in_sample_cpd[:, :, CENTER] = predictions_in_sample[:, :]
in_sample_cpd[:, :, 0] = -500
in_sample_cpd[:, :, -1] = 4000

quantiles = np.zeros((N_TEST, H, len(ALPHA)))
intervals = np.zeros((N_TEST, H, len(Q) + 2))

intervals[:, :, CENTER] = predictions[CALIB_SET:, :]
intervals[:, :, 0] = -500
intervals[:, :, -1] = 4000


# %%
for (a, alpha), h in tqdm(product(enumerate(ALPHA), range(H))):
    start = time.time()
    quantiles[:, h, a] = aci_clipped(
        scores[:, 0],
        alpha=alpha,
        lr=0.01,
        window_length=1000,
        T_burnin=CALIB_SET,
        ahead=1,
    )["q"][CALIB_SET:]
    intervals[:, h, CENTER + (a + 1)] = predictions[CALIB_SET:, h] + quantiles[:, h, a]
    intervals[:, h, CENTER - (a + 1)] = predictions[CALIB_SET:, h] - quantiles[:, h, a]
    stop = time.time()
    timings[CALIB_SET:, H] += (stop - start) / N_TEST


# %%
# Evaluation of the predictions
sims = np.full((N_TEST, 1, N_SIMS, H), np.nan, dtype=np.float32)

quantpred = intervals[:, :, 1:-1]
error_mean = prices_test - predictions[CALIB_SET:, :]
error_mae = prices_test - predictions[CALIB_SET:, :]
error_crps = sr.crps_quantile(
    prices_test,
    quantpred,
    Q,
)

jpb = intervals[:, :, np.isclose(QQ, 0.025) | np.isclose(QQ, 0.975)]
MODEL_NAMES = np.array(["ar_cp"])

np.savez(
    os.path.join(FOLDER_RESULTS, "benchmark_cp.npz"),
    predictions=predictions[CALIB_SET:, :],
    intervals=intervals[CALIB_SET:, :, :],
    quantiles=quantiles,
    error_mean=error_mean,
    error_med=error_mae,
    error_crps=error_crps,
    jpb=jpb,
    simulations=sims,
    model_names=MODEL_NAMES,
    timings=timings[CALIB_SET:, :],
    residuals=residuals,
)


# %%
# Transformation of the predictions via the CPD does not work well
# Since the 99 percentiles are to coarse and interpolation to the -500 and 4000
# max and min prices of the exchange leads to very bad results.

# Also using different interpolation methods does not help
# We will use the quantiles to get the in-sample CPD


# %%
# Get the in-sample CPD
# We use the quantiles to get the in-sample CPD
# for i, n in enumerate(range(N_TRAIN, N)):
#     if i == 0:
#         in_sample_cpd[:n, :, 1:CENTER] = in_sample_cpd[
#             :n, :, CENTER, None
#         ] - np.expand_dims(quantiles[i, :, ::-1], 0)
#         in_sample_cpd[:n, :, CENTER + 1 : -1] = in_sample_cpd[
#             :n, :, CENTER, None
#         ] + np.expand_dims(quantiles[i, :, :], 0)
#     else:
#         in_sample_cpd[n - 1, :, 1:CENTER] = in_sample_cpd[
#             n - 1, :, CENTER, None
#         ] - np.expand_dims(quantiles[i, :, ::-1], 0)
#         in_sample_cpd[n - 1, :, CENTER + 1 : -1] = in_sample_cpd[
#             n - 1, :, CENTER, None
#         ] + np.expand_dims(quantiles[i, :, :], 0)


# # %%
# # Transform the predictions in-sample via PIT to uniform space

# QQ = np.concatenate(([0], Q, [1]))

# uniform = np.zeros_like(predictions_in_sample)
# for i, h in tqdm(product(range(N - 1), range(H))):
#     uniform[i, h] = np.interp(
#         x=prices[i, h],
#         xp=in_sample_cpd[i, h, :],
#         fp=QQ,
#     )

# # %% Copula modelling on top of the CPD
# # Same approach as with the marginal
# copula = {}
# copula[0] = OnlineGaussianCopula()
# copula[1] = OnlineSparseGaussianCopula()

# N_COPULA = len(copula)
# timings_copula = np.zeros((N_TEST, N_COPULA))
# simulations_copula = np.empty([N_TEST, N_COPULA + 1, N_SIMS, H])
# predictions_loc_copula = np.zeros((N_TEST, N_COPULA, H))
# predictions_cov_copula = np.zeros((N_TEST, N_COPULA, H, H))

# # For numerical robustness
# # Avoid inf at the CDF of 0 and 1
# SMALL_VALUE = 1e-6
# uniform = np.clip(uniform, SMALL_VALUE, 1 - SMALL_VALUE)

# for t, i in tqdm(enumerate(range(N_TRAIN, N))):
#     if t == 0:
#         start = time.time()
#         copula[0].fit(uniform[:i, :])
#         stop = time.time()
#         timings_copula[t, 0] = stop - start

#         start = time.time()
#         copula[1].fit(uniform[:i, :])
#         stop = time.time()
#         timings_copula[t, 1] = stop - start

#     else:
#         start = time.time()
#         copula[0].update(uniform[[i - 1], :])
#         stop = time.time()
#         timings_copula[t, 0] = stop - start

#         start = time.time()
#         copula[1].update(uniform[[i - 1], :])
#         stop = time.time()
#         timings_copula[t, 1] = stop - start

#     predictions_loc_copula[t, 0] = copula[0].loc
#     predictions_cov_copula[t, 0] = copula[0].cov

#     predictions_loc_copula[t, 1] = copula[1].loc
#     predictions_cov_copula[t, 1] = copula[1].opt_cov

#     # Simulations from the marginal only
#     for h in range(H):
#         simulations_copula[t, 0, ..., h] = np.interp(
#             x=np.random.uniform(size=N_SIMS),
#             fp=intervals[t, h, :],
#             xp=QQ,
#         )

#     # Simulation from the copula
#     for c in copula.keys():
#         samples_uniform = copula[c].sample(N_SIMS)  # Need to make this pass-able
#         samples_margin = np.zeros_like(samples_uniform)
#         for h in range(H):
#             simulations_copula[t, c + 1, ..., h] = np.interp(
#                 x=samples_uniform[:, h],
#                 fp=intervals[t, h, :],
#                 xp=QQ,
#             )

# # This does not work.
# # %%
# # %%
