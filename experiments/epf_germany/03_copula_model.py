# ENSURE THAT YOU HAVE THE SRC CODE ON THE PATH
import sys

sys.path.insert(1, "../../../online_mv_distreg/")

import time

import numpy as np
import pandas as pd
import rolch
from const_and_helper import N_SIMS, H
from tqdm import tqdm

from src.estimator.online_copula import OnlineGaussianCopula, OnlineSparseGaussianCopula

print("ROLCH:", rolch.__version__)

np.set_printoptions(precision=3, suppress=True)

df_X = pd.read_csv("prepared_x.csv", index_col=0)
df_y = pd.read_csv("prepared_y.csv", index_col=0)
df_X["constant"] = 1

# Get the indices
IDX_TRAIN = df_X.flag.to_numpy() == "train"
IDX_TEST = df_X.flag.to_numpy() == "test"

N_TRAIN = np.sum(IDX_TRAIN)
N_TEST = np.sum(IDX_TEST)
N = df_X.shape[0]

# Define the variables
X = df_X.drop(["flag"], axis=1)
y = df_y

y_numpy = y.values
X_numpy = X.values


# Get the univariate predictions
univariate_file = np.load("results/univariate_predictions_distreg.npz")
univariate_prediction = univariate_file["predictions_outofsample"]
univariate_insample = univariate_file["predictions_insample"]
univariate_timings = univariate_file["timings"]

# Make the copula model
copula = {}
copula[0] = OnlineGaussianCopula()
copula[1] = OnlineSparseGaussianCopula()

N_COPULA = len(copula)
RANDOM_STATE = 123

simulations_copula = np.empty([N_TEST, N_COPULA + 1, N_SIMS, H])
margin_distribution = rolch.DistributionT()
timings_copula = np.zeros((N_TEST, N_COPULA))

uniform = np.zeros((N - 1, H))
for h in range(H):
    uniform[:, h] = margin_distribution.cdf(
        y_numpy[:-1, h],
        univariate_insample[:, h, :],
    )

# For numerical robustness
# Avoid inf at the CDF of 0 and 1
SMALL_VALUE = 1e-6
uniform = np.clip(uniform, SMALL_VALUE, 1 - SMALL_VALUE)

predictions_loc_copula = np.zeros((N_TEST, N_COPULA, H))
predictions_cov_copula = np.zeros((N_TEST, N_COPULA, H, H))

print("Runnning Forecasting study for Copula Model")
# Run the forecasting study
for t, i in tqdm(enumerate(range(N_TRAIN, N))):
    if t == 0:
        start = time.time()
        copula[0].fit(uniform[:i, :])
        stop = time.time()
        timings_copula[t, 0] = stop - start

        start
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
        theta=univariate_prediction[t, :, :], size=N_SIMS
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


prediction_uniform = np.zeros((N_TEST, H))
for h in range(H):
    prediction_uniform[:, h] = rolch.DistributionT().cdf(
        y=y_numpy[IDX_TEST, h], theta=univariate_prediction[:, h, :]
    )

# Save the results
np.savez_compressed(
    file="results/pred_copula.npz",
    predictions_cov=predictions_cov_copula,
    predictions_loc=predictions_loc_copula,
    predictions_uni=prediction_uniform,
    timings=timings_copula,
)

np.savez_compressed(
    file="results/sims_copula.npz",
    simulations=simulations_copula,
)
