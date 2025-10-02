# %%
# Allow running as notebook and as script
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# %%
import copy
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from const_and_helper import (  # noqa: E501
    FOLDER_DATA,
    FOLDER_FIGURES,
    FOLDER_RESULTS,
    FORGET,
    PLT_SAVE_OPTIONS,
    PLT_TEX_OPTIONS,
    RANDOM_STATE,
    H,
    get_cholesky_data,
    get_daily_data_index,
    get_hourly_mean_model_data_index,
)
from ondil.distributions import MultivariateStudentTInverseModifiedCholesky
from ondil.estimators import MultivariateOnlineDistributionalRegressionPath
from ondil.information_criteria import InformationCriterion
from ondil.links import (  # noqa: E501
    Identity,
    InverseSoftPlusShiftTwo,
    InverseSoftPlusShiftValue,
    Log,
    LogShiftTwo,
    MatrixDiag,
    MatrixDiagTril,
    Sqrt,
)

plt.rcParams.update(PLT_TEX_OPTIONS)
np.set_printoptions(precision=3, suppress=True)

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

scaling_m = scaling_m.values.astype(np.float32)[IDX_TRAIN]
scaling_s = scaling_s.values.astype(np.float32)[IDX_TRAIN]
diag_s = np.zeros((N_TRAIN, H, H))
diag_s[:, range(H), range(H)] = scaling_s


prices = pd.read_csv(os.path.join(FOLDER_DATA, "prices_untransformed.csv"), index_col=0)
prices = prices.values.astype(np.float32)
prices_test = prices[IDX_TEST, :]
prices_train = prices[IDX_TRAIN, :]

# %%
# Set the max rank for the LRA matrix
dof_guesstimate = 10
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


# %%

MAX_SIZE = 11
VERBOSE = 0
TO_SCALE = X.loc[:, :].columns.str.contains("load|gas|res|coal|eua")

rel_tol = 0.001
abs_tol = rel_tol * N_TRAIN

general_params = dict(
    ic="bic",
    forget=FORGET,
    scale_inputs=TO_SCALE,
    verbose=VERBOSE,
    early_stopping=False,  # important
    max_regularisation_size=MAX_SIZE,  # we do not stop the models from growing
    max_iterations_inner=30,
    max_iterations_outer=10,
    rel_tol_inner=rel_tol,
    rel_tol_outer=rel_tol,
    abs_tol_inner=abs_tol,
    abs_tol_outer=abs_tol,
)

params_mcd = general_params | dict(
    dampen_estimation={0: False, 1: True, 2: True, 3: True},
)

estimator = {}
base_model = MultivariateOnlineDistributionalRegressionPath(
    distribution=distribution_mcd_t,
    equation=equation_mcd_t,
    method="ols",
    **params_mcd,
)

# %%
N_FOLDS = 8
FOLD_SIZE = 100

generator = np.random.default_rng(RANDOM_STATE)
IDX_FOLDS = generator.choice(
    np.arange(N_TRAIN), size=N_FOLDS * FOLD_SIZE, replace=False
)
IDX_CV = np.full((N_TRAIN), N_FOLDS + 1)
IDX_CV[IDX_FOLDS] = np.repeat(np.arange(N_FOLDS), FOLD_SIZE)

X_train = X_numpy[IDX_TRAIN, :]
y_train = y_numpy[IDX_TRAIN, :]

# %%
for fold in range(N_FOLDS):
    print(f"Fold {fold + 1} of {N_FOLDS}")
    cv_index_train = IDX_CV != fold
    estimator[fold] = copy.copy(base_model)
    estimator[fold].fit(
        X=X_train[cv_index_train, :],
        y=y_train[cv_index_train, :],
    )

# %%
log_scores = np.zeros((N_FOLDS, N_TRAIN, MAX_SIZE))
for f in range(N_FOLDS):
    pred = estimator[f].predict_all_adr(X_train[:, :])
    for a in pred.keys():
        predictions = distribution_mcd_t.theta_to_scipy(pred[a])
        location = predictions["loc"] * scaling_s + scaling_m
        scale = diag_s @ predictions["shape"] @ diag_s.swapaxes(-1, -2)
        df = predictions["df"].squeeze()
        for t in range(N_TRAIN):
            log_scores[f, t, a] = -st.multivariate_t(
                loc=location[t, :],
                shape=scale[t, :, :],
                df=df[t],
            ).logpdf(
                prices_train[t, :],
            )

# %%
# Calculate the average scores
# Calculate standard deviation of log-scores
avg_scores = np.zeros((N_FOLDS, 2, MAX_SIZE))
for f in range(N_FOLDS):
    avg_scores[f, 0] = log_scores[f][IDX_CV == f].mean(0)
    avg_scores[f, 1] = log_scores[f][IDX_CV != f].mean(0)


# %%
# Calculate 90% confidence bands (5th and 95th percentiles) across folds for each model
z = st.norm().ppf(1 - 0.05)
lower_band = avg_scores.mean(0) - z * avg_scores.std(0)
upper_band = avg_scores.mean(0) + z * avg_scores.std(0)

nonzero_coefficients = np.zeros((N_FOLDS, 11))
for f, a in product(range(N_FOLDS), range(MAX_SIZE)):
    nonzero_coefficients[f, a] = estimator[f].count_nonzero_coef(estimator[f].coef_, a)

# %%
np.savez_compressed(
    file=os.path.join(FOLDER_RESULTS, "overfitting.npz"),
    avg_scores=avg_scores,
    lower_band=lower_band,
    upper_band=upper_band,
    log_scores=log_scores,
    nonzero_coefficients=nonzero_coefficients,
    IDX_CV=IDX_CV,
    N_FOLDS=N_FOLDS,
    FOLD_SIZE=FOLD_SIZE,
)

# %%
# Load the data
loaded = np.load(os.path.join(FOLDER_RESULTS, "overfitting.npz"))
avg_scores = loaded["avg_scores"]
lower_band = loaded["lower_band"]
upper_band = loaded["upper_band"]
log_scores = loaded["log_scores"]
nonzero_coefficients = loaded["nonzero_coefficients"]
IDX_CV = loaded["IDX_CV"]
N_FOLDS = int(loaded["N_FOLDS"])
FOLD_SIZE = int(loaded["FOLD_SIZE"])


# %%
# Calculate IC-based model choice
# We use the out-of-sample log-likelihood for the folds
# (i.e., the in-sample log-likelihood on the validation fold)
# and calculate the IC for each fold and each model size.
# We then take the majority vote across folds for each criterion.
# Note that the number of observations is N_TRAIN - FOLD_SIZE
criteria = ["aic", "hqc", "bic"]
fold_ic = np.zeros((len(criteria), N_FOLDS))
majority_ic = np.zeros(len(criteria))

for i, criterion in enumerate(criteria):
    ic = (
        InformationCriterion(
            n_observations=N_TRAIN - FOLD_SIZE,
            n_parameters=nonzero_coefficients,
            criterion=criterion,
        )
        .from_ll(
            log_likelihood=-avg_scores[:, 1, :] * (N_TRAIN - FOLD_SIZE),
        )
        .T
    )
    fold_ic[i,] = ic.argmin(0)
    majority_ic[i] = np.median(fold_ic[i,])

# %%
# Plot confidence bands
# plt.rcParams.update({"font.family": "serif", "text.usetex": True})
plt.figure(figsize=(8, 5))
plt.plot(
    avg_scores.mean(0).T,
    marker="o",
    label=["Out-of-sample", "In-sample"],
)
plt.fill_between(
    np.arange(avg_scores.shape[2]),
    lower_band[0, :],
    upper_band[0, :],
    color="gray",
    alpha=0.3,
)
plt.fill_between(
    np.arange(avg_scores.shape[2]),
    lower_band[1, :],
    upper_band[1, :],
    color="gray",
    alpha=0.3,
)
for i, criterion in enumerate(criteria):
    plt.axvline(
        majority_ic[i],
        color=f"C{i+2}",
        ls=":",
        label=f"{criterion.upper()} Choice ({int(majority_ic[i])} off-diags)",
    )
plt.grid(ls=":")

ax1 = plt.gca()
ax1.set_xlabel(r"\#{} of off-diagonals modelled")
ax1.set_ylabel("Log-Score")

ax2 = ax1.twinx()
ax2.plot(
    nonzero_coefficients.mean(0),
    color="tab:red",
    marker="o",
    label="Nonzero Coefficients",
)
ax2.set_ylabel("Number of Nonzero Coefficients", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 1.2),
)

ax1.set_xticks(np.arange(MAX_SIZE))

plt.tight_layout()
plt.savefig(os.path.join(FOLDER_FIGURES, "overfitting.pdf"), **PLT_SAVE_OPTIONS)
plt.savefig(os.path.join(FOLDER_FIGURES, "overfitting.png"), **PLT_SAVE_OPTIONS)
plt.show(block=False)

# %%
plt.tight_layout()
plt.savefig(os.path.join(FOLDER_FIGURES, "overfitting.pdf"), **PLT_SAVE_OPTIONS)
plt.savefig(os.path.join(FOLDER_FIGURES, "overfitting.png"), **PLT_SAVE_OPTIONS)
plt.show(block=False)

# %%
