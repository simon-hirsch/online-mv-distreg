# ENSURE THAT YOU HAVE THE SRC CODE ON THE PATH
import sys

sys.path.insert(1, "../../../online_mv_distreg/")

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
    get_low_rank_data_index,
    get_to_scale,
)
from tqdm import tqdm

from src.distribution import (
    MultivariateStudentTInverseCholesky,
    MultivariateStudentTInverseLowRank,
    get_v_indices,
)
from src.estimator import MultivariateOnlineDistributionalRegressionADRPath
from src.link import InverseSoftPlusLink, MatrixDiagLink, MatrixDiagTrilLink

print("ROLCH", rolch.__version__)

np.set_printoptions(precision=3, suppress=True)

df_X = pd.read_csv("prepared_x.csv", index_col=0)
df_y = pd.read_csv("prepared_y.csv", index_col=0)
df_X["constant"] = 1

# Prepare indices
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


# Set the max rank for the LRA matrix
rank = 2

loc_link = rolch.IdentityLink()
loc_link._valid_structures = ["matrix", "square_matrix", "vector"]

scale_link_1 = MatrixDiagLink(rolch.SqrtLink())
scale_link_2 = rolch.IdentityLink()
scale_link_2._valid_structures = ["matrix", "square_matrix", "vector"]

tail_link = rolch.LogShiftTwoLink()
tail_link._valid_structures = ["matrix", "square_matrix", "vector"]


# Distribution objects
distribution_lr_t = MultivariateStudentTInverseLowRank(
    loc_link=loc_link,
    scale_link_1=scale_link_1,
    scale_link_2=scale_link_2,
    tail_link=tail_link,
    rank=rank,
    dof_guesstimate=1e6,
)

distribution_cd_t = MultivariateStudentTInverseCholesky(
    loc_link=loc_link,
    scale_link=MatrixDiagTrilLink(
        diag_link=InverseSoftPlusLink(),
        tril_link=rolch.IdentityLink(),
    ),
    tail_link=tail_link,
    dof_guesstimate=1e6,
)


# Define the mapping of k to i, j for the lower
# triangular Cholesky decomposition in the CD t-distribution.
mapping_t = np.tril_indices(H)

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

VERBOSE = 3
LAMBDA_N = 50
TO_SCALE = get_to_scale(X)

# Use N-R-Scoring by taking the
# partial derivatives wrt theta and the
# first and second derivatives of the link
# function.
GENERIC_SCORING = True

params_cd = dict(
    forget=0.0,
    scale_inputs=True,
    to_scale=TO_SCALE,
    verbose=VERBOSE,
    early_stopping=True,
    weight_delta={0: 1, 1: 1, 2: 1},
    max_iterations_inner=30,
    max_iterations_outer=10,
    dampen_estimation={0: False, 1: True, 2: False},
    lambda_n=LAMBDA_N,
    generic_scoring=GENERIC_SCORING,
)

params_lr = dict(
    forget=0.0,
    scale_inputs=True,
    to_scale=TO_SCALE,
    verbose=VERBOSE,
    early_stopping=True,
    weight_delta={0: 1, 1: 1, 2: 1, 3: 1},
    max_iterations_inner=30,
    max_iterations_outer=10,
    approx_fast_model_selection=False,
    dampen_estimation={0: False, 1: True, 2: False},
    fit_intercept={0: True, 1: True, 2: False, 3: True},
    lambda_n=LAMBDA_N,
    lambda_eps=0.01,
    generic_scoring=GENERIC_SCORING,
)


estimator = {}
estimator[0] = MultivariateOnlineDistributionalRegressionADRPath(
    distribution=distribution_cd_t,
    equation=equation_cd_t,
    max_regularisation_size=1,
    method="ols",
    **params_cd,
)

estimator[1] = MultivariateOnlineDistributionalRegressionADRPath(
    distribution=distribution_lr_t,
    equation=equation_lr_t,
    max_regularisation_size=1,
    method="ols",
    **params_lr,
)
estimator[2] = MultivariateOnlineDistributionalRegressionADRPath(
    distribution=distribution_cd_t,
    equation=equation_cd_t,
    max_regularisation_size=6,
    method="ols",
    **params_cd,
)
estimator[3] = MultivariateOnlineDistributionalRegressionADRPath(
    distribution=distribution_lr_t,
    equation=equation_lr_t,
    method="ols",
    **params_lr,
)
estimator[4] = MultivariateOnlineDistributionalRegressionADRPath(
    distribution=distribution_cd_t,
    equation=equation_cd_t,
    max_regularisation_size=6,
    method="lasso",
    **params_cd,
)
estimator[5] = MultivariateOnlineDistributionalRegressionADRPath(
    distribution=distribution_lr_t,
    equation=equation_lr_t,
    method="lasso",
    **params_lr,
)


# RUN THE SIMULATION STUDY!!
N_MODELS = len(estimator)

timings = np.zeros([N_MODELS, N_TEST])
predictions_loc = np.zeros((N_TEST, N_MODELS, H))
predictions_cov = np.zeros((N_TEST, N_MODELS, H, H))
predictions_dof = np.zeros((N_TEST, N_MODELS, 1))

# for m in range(N_MODELS):
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

            pred = estimator[m].predict(X=X_numpy[[i], :])
            pred = estimator[m].distribution.theta_to_scipy(pred)

            predictions_loc[k, m, ...] = pred["loc"].squeeze()
            predictions_cov[k, m, ...] = pred["shape"].squeeze()
            predictions_dof[k, m, ...] = pred["dof"].squeeze()

            np.savez_compressed(
                file="results/pred_multivariate.npz",
                timings=timings,
                predictions_loc=predictions_loc,
                predictions_cov=predictions_cov,
                predictions_dof=predictions_dof,
            )
    except Exception as e:
        print("###############################################################")
        print(f"Model {m}, step {k, i}, failed with exception", e)
        print("###############################################################")


# Create the simulation
RANDOM_STATE = 123
simulations = np.empty((N_TEST, N_MODELS, N_SIMS, H))

for m in range(N_MODELS):
    for t in range(N_TEST):
        try:
            simulations[t, m, ...] = st.multivariate_t(
                loc=predictions_loc[t, m],
                shape=predictions_cov[t, m],
                df=predictions_dof[t, m],
            ).rvs((N_SIMS, 1), random_state=RANDOM_STATE + t)
        except Exception as _:
            pass


# # Save everything to the
np.savez_compressed(
    file="results/pred_multivariate.npz",
    timings=timings,
    predictions_loc=predictions_loc,
    predictions_cov=predictions_cov,
    predictions_dof=predictions_dof,
)

np.savez_compressed(
    file="results/sims_multivariate.npz",
    simulations=simulations,
)
