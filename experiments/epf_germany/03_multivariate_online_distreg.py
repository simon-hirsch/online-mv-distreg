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
import scipy.stats as st
from const_and_helper import (  # noqa: E501
    FOLDER_DATA,
    FOLDER_RESULTS,
    FORGET,
    N_SIMS,
    RANDOM_STATE,
    H,
    get_cholesky_data,
    get_daily_data_index,
    get_hourly_mean_model_data_index,
    get_low_rank_data_index,
)
from const_and_helper.distribution import get_v_indices
from ondil.distributions import (
    MultivariateStudentTInverseCholesky,
    MultivariateStudentTInverseLowRank,
    MultivariateStudentTInverseModifiedCholesky,
)
from ondil.estimators import MultivariateOnlineDistributionalRegressionPath
from ondil.links import (  # noqa: E501
    Identity,
    InverseSoftPlus,
    InverseSoftPlusShiftTwo,
    InverseSoftPlusShiftValue,
    Log,
    LogShiftTwo,
    MatrixDiag,
    MatrixDiagTril,
    Sqrt,
)
from tqdm import tqdm

np.set_printoptions(precision=3, suppress=True)

df_X = pd.read_csv(os.path.join(FOLDER_DATA, "prepared_long_x.csv"), index_col=0)
df_y = pd.read_csv(os.path.join(FOLDER_DATA, "prepared_long_y.csv"), index_col=0)

X = df_X.drop(["flag"], axis=1)
X_numpy = X.values
y_numpy = df_y.values

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

# We need to shift the scaling_m and scaling_s by one day
# Because we need the inverse transformation of the to happen with the
# correct information set!
scaling_m = scaling_m.values.astype(np.float32)[np.roll(IDX_TEST, -1), :]
scaling_s = scaling_s.values.astype(np.float32)[np.roll(IDX_TEST, -1), :]

prices = pd.read_csv(os.path.join(FOLDER_DATA, "prices_untransformed.csv"), index_col=0)
prices = prices.values.astype(np.float32)
prices_test = prices[IDX_TEST, :]

# %%
# Set the max rank for the LRA matrix
rank = 2
dof_guesstimate = 10

# %%
# Distribution objects
distribution_lr_t = MultivariateStudentTInverseLowRank(
    loc_link=Identity(),
    scale_link_1=MatrixDiag(Sqrt()),
    scale_link_2=Identity(),
    tail_link=InverseSoftPlusShiftValue(2.1),
    rank=rank,
    dof_guesstimate=dof_guesstimate,
)

distribution_cd_t = MultivariateStudentTInverseCholesky(
    loc_link=Identity(),
    scale_link=MatrixDiagTril(
        diag_link=Log(),
        tril_link=Identity(),
    ),
    tail_link=InverseSoftPlusShiftValue(2.1),
    dof_guesstimate=dof_guesstimate,
)

distribution_mcd_t = MultivariateStudentTInverseModifiedCholesky(
    loc_link=Identity(),
    scale_link_1=MatrixDiag(
        diag_link=Log(),
    ),
    scale_link_2=MatrixDiagTril(
        diag_link=Identity(),
        tril_link=Identity(),
    ),
    tail_link=InverseSoftPlusShiftValue(2.1),
    dof_guesstimate=dof_guesstimate,
)

# Define the mapping of k to i, j for the lower
# triangular Cholesky decomposition in the CD t-distribution.
mapping_t = np.tril_indices(H)
mapping_t_mcd = np.tril_indices(H, k=-1)

# Define all equations for the models
# We have the same variables for the
# - Mean Models
# - Diagonals of the precision matrix
# For the LRA and the off-diagonal CD parameters, the models differ slightly.

equation_cd_t = {
    0: {
        h: np.arange(X.shape[1])[get_hourly_mean_model_data_index(X, h)]
        for h in range(H)
    },
    1: {
        k: np.arange(X.shape[1])[get_cholesky_data(X, h, i)]
        for k, (h, i) in enumerate(zip(*mapping_t))
    },
    2: {0: np.arange(X.shape[1])[get_daily_data_index(X)]},
}

equation_mcd_t = {
    0: {
        h: np.arange(X.shape[1])[get_hourly_mean_model_data_index(X, h, linear=True)]
        for h in range(H)
    },
    1: {
        k: np.arange(X.shape[1])[get_cholesky_data(X, h, h)]
        for k, h in enumerate(range(H))
    },
    2: {
        k: np.arange(X.shape[1])[get_cholesky_data(X, i, j)]
        for k, (i, j) in enumerate(zip(*mapping_t_mcd))
    },
    3: {0: np.arange(X.shape[1])[get_daily_data_index(X)]},
}

equation_lr_t = {
    0: {
        h: np.arange(X.shape[1])[get_hourly_mean_model_data_index(X, h)]
        for h in range(H)
    },
    1: {h: np.arange(X.shape[1])[get_cholesky_data(X, h, h)] for h in range(H)},
    2: {
        k: np.arange(X.shape[1])[get_low_rank_data_index(X, h, r)]
        for k, (h, r) in enumerate(zip(*get_v_indices(H, distribution_lr_t.rank)))
    },
    3: {0: np.arange(X.shape[1])[get_daily_data_index(X)]},
}

# %%

VERBOSE = 3
TO_SCALE = X.loc[:, :].columns.str.contains("load|gas|res|coal|eua")

rel_tol = 0.001
abs_tol = 0.001

general_params = dict(
    ic="bic",
    early_stopping_criteria="bic",
    forget=FORGET,
    scale_inputs=TO_SCALE,
    verbose=VERBOSE,
    early_stopping=True,
    max_iterations_inner=30,
    max_iterations_outer=10,
    rel_tol_inner=rel_tol,
    rel_tol_outer=rel_tol,
    abs_tol_inner=abs_tol,
    abs_tol_outer=abs_tol,
    early_stopping_rel_tol=0.01,
    early_stopping_abs_tol=0.01,
)

params_cd = general_params | dict(
    dampen_estimation={0: False, 1: True, 2: True},
)

params_mcd = general_params | dict(
    dampen_estimation={0: False, 1: True, 2: True, 3: True},
)

params_lr = general_params | dict(
    approx_fast_model_selection=False,
    dampen_estimation={0: False, 1: True, 2: True, 3: True},
    fit_intercept={0: True, 1: True, 2: False, 3: True},
    # We have noticed that the LRA models sometimes need a bit more tolerance
    # to converge, especially the lasso ones
    # otherwise convergence is excessively slow
    abs_tol_inner=0.1,
    abs_tol_outer=0.1,
)

# Model Ordering
# 3 x CD t-distribution
# 3 x MCD t-distribution
# 3 x LR t-distribution

# Cholesky t-distribution
estimator = {}
estimator[0] = MultivariateOnlineDistributionalRegressionPath(
    distribution=distribution_cd_t,
    equation=equation_cd_t,
    max_regularisation_size=1,
    method="ols",
    **params_cd,
)
estimator[1] = MultivariateOnlineDistributionalRegressionPath(
    distribution=distribution_cd_t,
    equation=equation_cd_t,
    max_regularisation_size=6,
    method="ols",
    **params_cd,
)
estimator[2] = MultivariateOnlineDistributionalRegressionPath(
    distribution=distribution_cd_t,
    equation=equation_cd_t,
    max_regularisation_size=6,
    method="lasso",
    **params_cd,
)

# Modified Cholesky t-distribution
estimator[3] = MultivariateOnlineDistributionalRegressionPath(
    distribution=distribution_mcd_t,
    equation=equation_mcd_t,
    max_regularisation_size=1,
    method="ols",
    **params_mcd,
)
estimator[4] = MultivariateOnlineDistributionalRegressionPath(
    distribution=distribution_mcd_t,
    equation=equation_mcd_t,
    max_regularisation_size=6,
    method="ols",
    **params_mcd,
)
estimator[5] = MultivariateOnlineDistributionalRegressionPath(
    distribution=distribution_mcd_t,
    equation=equation_mcd_t,
    max_regularisation_size=6,
    method="lasso",
    **params_mcd,
)

# Low Rank t-distribution
estimator[6] = MultivariateOnlineDistributionalRegressionPath(
    distribution=distribution_lr_t,
    equation=equation_lr_t,
    max_regularisation_size=1,
    method="ols",
    **params_lr,
)
estimator[7] = MultivariateOnlineDistributionalRegressionPath(
    distribution=distribution_lr_t,
    equation=equation_lr_t,
    method="ols",
    **params_lr,
)
estimator[8] = MultivariateOnlineDistributionalRegressionPath(
    distribution=distribution_lr_t,
    equation=equation_lr_t,
    method="lasso",
    **params_lr,
)

# %%
MODEL_NAMES = [
    "cd_ols_ind",
    "cd_ols_dep",
    "cd_lasso_dep",
    "mcd_ols_ind",
    "mcd_ols_dep",
    "mcd_lasso_dep",
    "lr_ols_ind",
    "lr_ols_dep",
    "lr_lasso_dep",
]


# %%
# RUN THE SIMULATION STUDY!!
N_MODELS = len(estimator)

timings = np.zeros([N_MODELS, N_TEST])
predictions_loc = np.zeros((N_TEST, N_MODELS, H))
predictions_cov = np.zeros((N_TEST, N_MODELS, H, H))
predictions_dof = np.zeros((N_TEST, N_MODELS, 1))
optimal_adr = np.zeros((N_TEST, N_MODELS))

# %%
# for m in range(4, 5):
for m in range(N_MODELS):
    try:
        print("###############################################################")
        print(f"Fitting Model {m}", N_TRAIN, N)
        print("###############################################################")
        for k, i in tqdm(enumerate(range(N_TRAIN, N))):
            if k == 0:
                start = time.time()
                estimator[m].fit(X=X_numpy[:i, :], y=y_numpy[:i, :])
                stop = time.time()
                timings[m, k] = stop - start

                # Silence estimators after first initial fit
                estimator[m].verbose = 0
            else:
                start = time.time()
                estimator[m].update(X=X_numpy[[i - 1], :], y=y_numpy[[i - 1], :])
                stop = time.time()
                timings[m, k] = stop - start

            pred = estimator[m].predict_distribution_parameters(X=X_numpy[[i], :])
            pred = estimator[m].distribution.theta_to_scipy(pred)

            predictions_loc[k, m, ...] = pred["loc"].squeeze()
            predictions_cov[k, m, ...] = pred["shape"].squeeze()
            # TODO this is inconsistent in the distribtutions,
            # Check what is correct in scipy!!
            if "df" in pred.keys():
                predictions_dof[k, m, ...] = pred["df"].squeeze()
            if "dof" in pred.keys():
                predictions_dof[k, m, ...] = pred["dof"].squeeze()

            optimal_adr[k, m] = estimator[m].optimal_adr_

    except Exception as e:
        print("###############################################################")
        print(f"Model {m}, step {k, i}, failed with exception", e)
        print("###############################################################")

# %%
# Re-scale the predictions to the original scale
location = np.zeros_like(predictions_loc)
scale = np.zeros_like(predictions_cov)
tail = np.zeros_like(predictions_dof)

location = np.zeros_like(predictions_loc)
scale = np.zeros_like(predictions_cov)

diag_s = np.zeros((N_TEST, H, H))
diag_s[:, range(H), range(H)] = scaling_s
for m in range(N_MODELS):
    location[:, m, :] = predictions_loc[:, m, :] * scaling_s + scaling_m
    scale[:, m, :] = diag_s @ predictions_cov[:, m] @ diag_s.swapaxes(-1, -2)
    tail[:, m, :] = predictions_dof[:, m, :]


# %% Save everything
np.savez_compressed(
    file=os.path.join(FOLDER_RESULTS, "pred_multivariate.npz"),
    model_names=MODEL_NAMES,
    timings=timings,
    predictions_loc_scaled=predictions_loc,
    predictions_cov_scaled=predictions_cov,
    predictions_dof_scaled=predictions_dof,
    predictions_loc=location,
    predictions_cov=scale,
    predictions_dof=tail,
    optimal_adr=optimal_adr,
)
# # %%
# # Load all necessary variables for scenario generation
# scenario_inputs = np.load(os.path.join(FOLDER_RESULTS, "pred_multivariate.npz"))
# location = scenario_inputs["predictions_loc"]
# scale = scenario_inputs["predictions_cov"]
# tail = scenario_inputs["predictions_dof"]

# %%
# Create the simulation
simulations = np.empty((N_TEST, N_MODELS, N_SIMS, H))
for m, t in product(range(N_MODELS), range(N_TEST)):
    try:
        simulations[t, m, ...] = st.multivariate_t(
            loc=location[t, m],
            shape=scale[t, m],
            df=tail[t, m],
        ).rvs((N_SIMS, 1), random_state=RANDOM_STATE + t)
    except Exception as _:
        pass

np.savez_compressed(
    file=os.path.join(FOLDER_RESULTS, "sims_multivariate.npz"),
    simulations=simulations.astype(np.float32),
    model_names=MODEL_NAMES,
)

# %%
