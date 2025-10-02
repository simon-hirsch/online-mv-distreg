# %%
# Allow running as notebook and as script
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# %%


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from const_and_helper import (  # noqa: E501
    FOLDER_DATA,
    FOLDER_FIGURES,
    FOLDER_RESULTS,
    PLT_SAVE_OPTIONS,
    PLT_TEX_OPTIONS,
    H,
    get_cholesky_data,
    get_daily_data_index,
    get_hourly_mean_model_data_index,
)
from ondil.distributions import MultivariateStudentTInverseModifiedCholesky
from ondil.estimators import MultivariateOnlineDistributionalRegressionPath
from ondil.links import (  # noqa: E501
    Identity,
    InverseSoftPlusShiftValue,
    Log,
    MatrixDiag,
    MatrixDiagTril,
)
from tqdm import tqdm

# Set options
plt.rcParams.update(PLT_TEX_OPTIONS)
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

dates = df_X.index

# %%
# for scaling/transformation
# y is already transformed
scaling_m = pd.read_csv(os.path.join(FOLDER_DATA, "prices_means.csv"), index_col=0)
scaling_s = pd.read_csv(os.path.join(FOLDER_DATA, "prices_variances.csv"), index_col=0)
scaling_m = scaling_m.values.astype(np.float32)
scaling_s = scaling_s.values.astype(np.float32)

prices = pd.read_csv(os.path.join(FOLDER_DATA, "prices_untransformed.csv"), index_col=0)
prices = prices.values.astype(np.float32)
prices_test = prices[IDX_TEST, :]

# %%
# Define the distribution and the model equation
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

mapping_t_mcd = np.tril_indices(H, k=-1)

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
VERBOSE = 3
FORGETS = np.array([0] + [1 / (365 * i) for i in range(5, 0, -1)])
TO_SCALE = X.loc[:, :].columns.str.contains("load|gas|res|coal|eua")

rel_tol = 0.001
abs_tol = 0.001

general_params = dict(
    ic="bic",
    early_stopping_criteria="bic",
    scale_inputs=TO_SCALE,
    max_iterations_inner=30,
    max_iterations_outer=10,
    rel_tol_inner=rel_tol,
    rel_tol_outer=rel_tol,
    abs_tol_inner=abs_tol,
    abs_tol_outer=abs_tol,
    early_stopping_rel_tol=0.01,
    early_stopping_abs_tol=0.01,
)

params_mcd = general_params | dict(
    dampen_estimation={0: False, 1: True, 2: True, 3: True},
)

# ADR with forget=0 and early_stopping=False
# We are interested in the performance of the
params_adr = params_mcd | {
    "early_stopping": False,
    "max_regularisation_size": 6,
    "forget": 0.0,
}

estimator_adr = MultivariateOnlineDistributionalRegressionPath(
    distribution=distribution_mcd_t,
    equation=equation_mcd_t,
    method="ols",
    verbose=1,
    **params_adr,
)

# %%
N_MODELS_ADR = params_adr["max_regularisation_size"]
logscores_adr = np.zeros([N_TEST, N_MODELS_ADR])
predictions_loc_adr = np.zeros((N_TEST, N_MODELS_ADR, H))
predictions_cov_adr = np.zeros((N_TEST, N_MODELS_ADR, H, H))
predictions_dof_adr = np.zeros((N_TEST, N_MODELS_ADR, 1))

# %%
# Iteratively update the model with the training data
# Save the log-scores and predictions
step = 1
for i, k in tqdm(enumerate(range(N_TRAIN, N, step))):
    if i == 0:
        estimator_adr.fit(X=X_numpy[:N_TRAIN, :], y=y_numpy[:N_TRAIN, :])
    else:
        estimator_adr.update(X=X_numpy[IDX_PRED, :], y=y_numpy[IDX_PRED, :])  # noqa

    IDX_PRED = np.arange(k, min(k + step, N))
    pred_adr = estimator_adr.predict_all_adr(X=X_numpy[IDX_PRED, :])

    for m, pred in pred_adr.items():
        pred_scipy = estimator_adr.distribution.theta_to_scipy(pred)

        # Scaling
        diag_s = np.zeros((len(IDX_PRED), H, H))
        diag_s[:, range(H), range(H)] = scaling_s[None, k - 1, :]
        mat_l = scaling_m[None, k - 1, :]
        mat_s = scaling_s[None, k - 1, :]

        predictions_loc_adr[IDX_PRED - N_TRAIN, m, ...] = (
            pred_scipy["loc"] * mat_s + mat_l
        )
        predictions_cov_adr[IDX_PRED - N_TRAIN, m, ...] = (
            diag_s @ pred_scipy["shape"] @ diag_s.swapaxes(-1, -2)
        )
        predictions_dof_adr[IDX_PRED - N_TRAIN, m, ...] = pred_scipy["df"]
        # Log-scores
        logscores_adr[i, m] = -st.multivariate_t(
            loc=predictions_loc_adr[i, m, :],
            shape=predictions_cov_adr[i, m, :, :],
            df=predictions_dof_adr[i, m, 0],
        ).logpdf(prices_test[i, :])

# %%
np.savez(
    os.path.join(FOLDER_RESULTS, "ablation_adr.npz"),
    logscores=logscores_adr,
    predictions_loc=predictions_loc_adr,
    predictions_cov=predictions_cov_adr,
    predictions_dof=predictions_dof_adr,
)

# %%
# file = np.load(
#     os.path.join(FOLDER_RESULTS, "ablation_adr.npz"),
# )

# logscores_adr = file["logscores"]
# predictions_loc_adr = file["predictions_loc"]
# predictions_cov_adr = file["predictions_cov"]
# predictions_dof_adr = file["predictions_dof"]

# %%
plt.figure(figsize=(5, 4))
plt.title(r"Effect of different regularization sizes $\alpha$")
bars = plt.bar(
    np.arange(N_MODELS_ADR),
    logscores_adr.mean(0),
    color="teal",
    edgecolor="black",
)
plt.xlabel(r"\# of off-diagonals modelled")
plt.ylabel("Log-Score")

plt.ylim(71, 99)
plt.yticks(np.linspace(71, 99, 9))
plt.xticks(np.arange(N_MODELS_ADR), fontsize=12)

for bar, score in zip(bars, logscores_adr.mean(0)):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{score:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

plt.gca().set_axisbelow(True)
plt.grid(axis="y", ls=":")
plt.tight_layout()
plt.savefig(
    os.path.join(FOLDER_FIGURES, "ablation_regularization.pdf"),
    **PLT_SAVE_OPTIONS,
)
plt.show(block=False)
