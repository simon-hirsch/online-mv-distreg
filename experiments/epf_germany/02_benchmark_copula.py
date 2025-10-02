# %%
# Allow running as notebook and as script
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# %%
import time

import numpy as np
import pandas as pd
from const_and_helper import FOLDER_DATA, FOLDER_RESULTS, N_SIMS, H
from const_and_helper.distribution import (
    OnlineGaussianCopula,
    OnlineSparseGaussianCopula,
)
from ondil.distributions import StudentT
from tqdm import tqdm

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
prices = pd.read_csv(os.path.join(FOLDER_DATA, "prices_untransformed.csv"), index_col=0)
prices = prices.values.astype(np.float32)
prices_test = prices[IDX_TEST, :]

# %%
# Get the univariate predictions
univariate_file = np.load(
    os.path.join(FOLDER_RESULTS, "univariate_predictions_distreg.npz")
)
univariate_prediction = univariate_file["predictions_outofsample"]
univariate_insample = univariate_file["predictions_insample"]
univariate_timings = univariate_file["timings"]

# Make the copula model
copula = {}
copula[0] = OnlineGaussianCopula()
copula[1] = OnlineSparseGaussianCopula()

N_COPULA = len(copula)

simulations_copula = np.empty([N_TEST, N_COPULA + 1, N_SIMS, H])
margin_distribution = StudentT()
timings_copula = np.zeros((N_TEST, N_COPULA))

uniform = np.zeros((N - 1, H))
for h in range(H):
    uniform[:, h] = margin_distribution.cdf(
        prices[:-1, h],
        univariate_insample[:, h, :],
    )

# For numerical robustness
# Avoid inf at the CDF of 0 and 1
SMALL_VALUE = 1e-6
uniform = np.clip(uniform, SMALL_VALUE, 1 - SMALL_VALUE)

predictions_loc_copula = np.zeros((N_TEST, N_COPULA, H))
predictions_cov_copula = np.zeros((N_TEST, N_COPULA, H, H))

# %%
# Run the forecasting study


for t, i in tqdm(enumerate(range(N_TRAIN, N))):
    if t == 0:
        start = time.time()
        copula[0].fit(uniform[:i, :])
        stop = time.time()
        timings_copula[t, 0] = stop - start

        start = time.time()
        copula[1].fit(uniform[:i, :])
        stop = time.time()
        timings_copula[t, 1] = stop - start

    else:
        start = time.time()
        copula[0].update(uniform[[i - 1], :])
        stop = time.time()
        timings_copula[t, 0] = stop - start

        start = time.time()
        copula[1].update(uniform[[i - 1], :])
        stop = time.time()
        timings_copula[t, 1] = stop - start

    predictions_loc_copula[t, 0] = copula[0].loc
    predictions_cov_copula[t, 0] = copula[0].cov

    predictions_loc_copula[t, 1] = copula[1].loc
    predictions_cov_copula[t, 1] = copula[1].opt_cov

    # Simulations from the marginal only
    simulations_copula[t, 0, ...] = margin_distribution.rvs(
        theta=univariate_prediction[t, :, :],
        size=N_SIMS,
    ).T

    # Simulation from the copula
    for c in copula.keys():
        samples_uniform = copula[c].sample(N_SIMS)  # Need to make this pass-able
        samples_margin = np.zeros_like(samples_uniform)
        for h in range(H):
            simulations_copula[t, c + 1, ..., h] = margin_distribution.ppf(
                samples_uniform[:, h],
                univariate_prediction[[t], h],
            )

# %%

MODEL_NAMES = ["no_copula", "gaussian_copula", "sparse_gaussian_copula"]

prediction_uniform = np.zeros((N_TEST, H))
for h in range(H):
    prediction_uniform[:, h] = margin_distribution.cdf(
        y=prices_test[:, h], theta=univariate_prediction[:, h, :]
    )

# Save the results
np.savez_compressed(
    file=os.path.join(FOLDER_RESULTS, "pred_copula.npz"),
    model_names=MODEL_NAMES,
    predictions_cov=predictions_cov_copula,
    predictions_loc=predictions_loc_copula,
    predictions_uni=prediction_uniform,
    timings=timings_copula,
)

np.savez_compressed(
    file=os.path.join(FOLDER_RESULTS, "sims_copula.npz"),
    model_names=MODEL_NAMES,
    simulations=simulations_copula,
)

# %%
