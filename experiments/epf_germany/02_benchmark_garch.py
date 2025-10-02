# %%
# Allow running as notebook and as script
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# %%
import time
from itertools import product

import arch
import numpy as np
import pandas as pd
import scipy.stats as st
from const_and_helper import FOLDER_DATA, FOLDER_RESULTS, N_SIMS, H
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

# Load the results
data = np.load(
    os.path.join(FOLDER_RESULTS, "pred_univariate_benchmark.npz"), allow_pickle=True
)
model_names = data["model_names"]
predictions_loc_scaled = data["predictions_loc_scaled"]
predictions_cov_scaled = data["predictions_cov_scaled"]
predictions_loc = data["predictions_loc"]
predictions_cov = data["predictions_cov"]
residuals = data["residuals"]
timings = data["timings"]


# %%
arch_variance_forecast = np.zeros((N_TEST, H))
arch_timings = np.copy(timings[:, 0])

for i, t in tqdm(enumerate(range(N_TRAIN, N))):
    start = time.time()
    for h in range(H):
        FIT_IDX = np.arange(i, N_TRAIN + i)
        model = arch.arch_model(residuals[FIT_IDX, h])
        model = model.fit(disp="off")
        arch_variance_forecast[i, h] = model.forecast(
            horizon=1,
            method="analytic",
        ).residual_variance.values.item()
    stop = time.time()
    arch_timings[i] += stop - start

# %%
# Scale the predictions
predictions_std = arch_variance_forecast**0.5 * scaling_s

# %%
np.savez_compressed(
    os.path.join(FOLDER_RESULTS, "pred_garch_benchmark.npz"),
    model_names=np.array(["ar_garch"]),
    predictions_std=predictions_std,
    predictions_loc=np.copy(predictions_loc),
    timings=arch_timings,
)

# %%
# Run simulations
simulations = np.zeros((N_TEST, 1, N_SIMS, H))
for t, h in product(range(N_TEST), range(H)):
    simulations[t, 0, :, h] = st.norm(
        loc=predictions_loc[t, h],
        scale=predictions_std[t, h],
    ).rvs(size=N_SIMS)
# %%
np.savez_compressed(
    os.path.join(FOLDER_RESULTS, "sims_garch_benchmark.npz"),
    model_names=np.array(["ar_garch"]),
    simulations=simulations,
)
# %%
