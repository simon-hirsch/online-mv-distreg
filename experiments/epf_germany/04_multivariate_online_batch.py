# %%
# Allow running as notebook and as script
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# %%
import pickle
import time
from copy import deepcopy

import matplotlib as mpl
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
    H,
    get_cholesky_data,
    get_daily_data_index,
    get_hourly_mean_model_data_index,
)
from const_and_helper.data_prep import get_batch_scaled_dataset
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from ondil.distributions import MultivariateStudentTInverseModifiedCholesky
from ondil.estimators import MultivariateOnlineDistributionalRegressionPath
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

params_mcd = general_params | dict(
    dampen_estimation={0: False, 1: True, 2: True, 3: True},
)

estimator = {}
base_model = MultivariateOnlineDistributionalRegressionPath(
    distribution=distribution_mcd_t,
    equation=equation_mcd_t,
    max_regularisation_size=5,
    method="ols",
    **params_mcd,
)

N_TEST_SHORT = 736
N_SHORT = N_TRAIN + N_TEST_SHORT


# %%
# Run initial fit
UPDATE_FREQS_ONLINE = np.array([1, 7, 14, 30, 60, 90, 180, 365])
UPDATE_FREQS_BATCH = np.array([7, 14, 30, 60, 90, 180, 365])

N_MODELS = len(UPDATE_FREQS_ONLINE) + (len(UPDATE_FREQS_BATCH) * 2)
UPDATE_FREQS = np.concatenate(
    [
        UPDATE_FREQS_ONLINE,
        UPDATE_FREQS_BATCH,
        UPDATE_FREQS_BATCH,
    ]
)


# %%
timings = np.zeros([N_TEST_SHORT, N_MODELS])
logscores = np.zeros([N_TEST_SHORT, N_MODELS])
predictions_loc = np.zeros((N_TEST_SHORT, N_MODELS, H))
predictions_cov = np.zeros((N_TEST_SHORT, N_MODELS, H, H))
predictions_dof = np.zeros((N_TEST_SHORT, N_MODELS, 1))


# %%
# Fit the base model to the initial training data
start = time.time()
base_model.fit(X=X_numpy[:N_TRAIN, :], y=y_numpy[:N_TRAIN, :])
stop = time.time()
base_model_train_time = stop - start

# %% Pickle the base model
with open(os.path.join(FOLDER_RESULTS, "base_model.pkl"), "wb") as f:
    pickle.dump(base_model, f)

# %%
# Iteratively update the model with the training data
# Save the log-scores and predictions
base_model.verbose = 0
for m, step in tqdm(enumerate(UPDATE_FREQS_ONLINE)):
    for i, k in enumerate(range(N_TRAIN, N_SHORT, step)):
        if i == 0:
            # Initial fit for all models
            estimator[m] = deepcopy(base_model)
            train_time = base_model_train_time
            estimator[m].verbose = 0
        else:
            start = time.time()
            estimator[m].update(X=X_numpy[IDX_PRED, :], y=y_numpy[IDX_PRED, :])  # noqa
            stop = time.time()
            train_time = stop - start

        IDX_PRED = np.arange(k, min(k + step, N_SHORT))

        pred = estimator[m].predict_distribution_parameters(X=X_numpy[IDX_PRED, :])
        pred_scipy = estimator[m].distribution.theta_to_scipy(pred)

        # Need to handle the scaling here!
        diag_s = np.zeros((len(IDX_PRED), H, H))
        diag_s[:, range(H), range(H)] = scaling_s[None, k - 1, :]
        mat_l = scaling_m[None, k - 1, :]
        mat_s = scaling_s[None, k - 1, :]

        predictions_loc[IDX_PRED - N_TRAIN, m, ...] = pred_scipy["loc"] * mat_s + mat_l
        predictions_cov[IDX_PRED - N_TRAIN, m, ...] = (
            diag_s @ pred_scipy["shape"] @ diag_s.swapaxes(-1, -2)
        )
        predictions_dof[IDX_PRED - N_TRAIN, m, ...] = pred_scipy["df"]

        timings[IDX_PRED - N_TRAIN, m] = train_time / step
        for t in IDX_PRED - N_TRAIN:
            logscores[t, m] = -st.multivariate_t(
                loc=predictions_loc[t, m, :],
                shape=predictions_cov[t, m, :, :],
                df=predictions_dof[t, m, 0],
            ).logpdf(prices_test[t, :])

# %%
# Now do the batch re-fitting
# This will take a loooooong time.
# We can be a bit smart and and don't re-estimate all the time.
# Nevertheless, we need to account for the time it would take
# to re-estimate the model.
# This is the subset of all the models that run on
# the same step size (30, 60, 90, ... days)

# EXPAND WINDOW VERSION
# KEEPING ALL THE DATA, TRAINING SET GROWS
for m, step in tqdm(enumerate(UPDATE_FREQS_BATCH)):
    print(step)
    mm = len(UPDATE_FREQS_ONLINE) + m
    SUBSET = UPDATE_FREQS_BATCH[UPDATE_FREQS_BATCH % step == 0]
    if np.all(np.isclose(logscores[:, mm], 0)):
        # Then we run the estimator
        for i, k in enumerate(range(N_TRAIN, N_SHORT, step)):
            try:
                k_start = max(0, k - N_TRAIN)
                k_start = 0
                IDX_TRAIN_BATCH = np.arange(k_start, k)
                DATES_BATCH = dates[IDX_TRAIN_BATCH]

                (
                    df_X_batch_scaled,
                    df_y_batch_scaled,
                    scaling_m_batch,
                    scaling_s_batch,
                ) = get_batch_scaled_dataset(DATES_BATCH)

                if i == 0:
                    # Initial fit for all models
                    estimator[mm] = deepcopy(base_model)
                    estimator[mm].verbose = 1
                    train_time = base_model_train_time
                else:
                    # Refit the model from scratch
                    start = time.time()
                    estimator[mm].fit(
                        # X=X_numpy[k_start:k, :],
                        # y=y_numpy[k_start:k, :],
                        X=df_X_batch_scaled.values[IDX_TRAIN_BATCH, :],
                        y=df_y_batch_scaled.values[IDX_TRAIN_BATCH, :],
                    )
                    stop = time.time()
                    train_time = stop - start

                # Now we do the predictions for all models in the subset
                # if the prediction step is a multiple of their step size
                for s in SUBSET:
                    if (k - N_TRAIN) % s == 0:
                        # the index in the predictions/logscores array
                        mmm = len(UPDATE_FREQS_ONLINE) + list(UPDATE_FREQS_BATCH).index(
                            s
                        )

                        IDX_PRED = np.arange(k, min(k + s, N_SHORT))

                        pred = estimator[mm].predict_distribution_parameters(
                            X=df_X_batch_scaled.values[IDX_PRED, :],
                        )
                        pred_scipy = estimator[mm].distribution.theta_to_scipy(pred)
                        diag_s = np.zeros((len(IDX_PRED), H, H))
                        diag_s[:, range(H), range(H)] = np.expand_dims(
                            scaling_s_batch.values, 0
                        )
                        mat_l = np.expand_dims(scaling_m_batch.values, 0)
                        mat_s = np.expand_dims(scaling_s_batch.values, 0)
                        predictions_loc[IDX_PRED - N_TRAIN, mmm, ...] = (
                            pred_scipy["loc"] * mat_s + mat_l
                        )
                        predictions_cov[IDX_PRED - N_TRAIN, mmm, ...] = (
                            diag_s @ pred_scipy["shape"] @ diag_s.swapaxes(-1, -2)
                        )
                        predictions_dof[IDX_PRED - N_TRAIN, mmm, ...] = pred_scipy["df"]
                        timings[IDX_PRED - N_TRAIN, mmm] = train_time / s

                        for t in IDX_PRED - N_TRAIN:
                            logscores[t, mmm] = -st.multivariate_t(
                                loc=predictions_loc[t, mmm, :],
                                shape=predictions_cov[t, mmm, :, :],
                                df=predictions_dof[t, mmm, 0],
                            ).logpdf(prices_test[t, :])

            except Exception as e:
                print(f"Error in model with step {step}: {e}, k={k}, i={i}")


# ROLLING WINDOW VERSION
# KEEPING THE SAME TRAINING SIZE
for m, step in tqdm(enumerate(UPDATE_FREQS_BATCH)):
    print(step)
    # In the rolling window case
    mm = len(UPDATE_FREQS_ONLINE) + len(UPDATE_FREQS_BATCH) + m
    SUBSET = UPDATE_FREQS_BATCH[UPDATE_FREQS_BATCH % step == 0]
    if np.all(np.isclose(logscores[:, mm], 0)):
        # Then we run the estimator
        for i, k in enumerate(range(N_TRAIN, N_SHORT, step)):
            try:
                # This is the main difference to the expanding window
                # We only keep the last N_TRAIN observations
                # for training
                k_start = max(0, k - N_TRAIN)
                IDX_TRAIN_BATCH = np.arange(k_start, k)
                DATES_BATCH = dates[IDX_TRAIN_BATCH]

                (
                    df_X_batch_scaled,
                    df_y_batch_scaled,
                    scaling_m_batch,
                    scaling_s_batch,
                ) = get_batch_scaled_dataset(DATES_BATCH)

                if i == 0:
                    # Initial fit for all models
                    estimator[mm] = deepcopy(base_model)
                    estimator[mm].verbose = 1
                    train_time = base_model_train_time
                else:
                    # Refit the model from scratch
                    start = time.time()
                    estimator[mm].fit(
                        # X=X_numpy[k_start:k, :],
                        # y=y_numpy[k_start:k, :],
                        X=df_X_batch_scaled.values[IDX_TRAIN_BATCH, :],
                        y=df_y_batch_scaled.values[IDX_TRAIN_BATCH, :],
                    )
                    stop = time.time()
                    train_time = stop - start

                for s in SUBSET:
                    if (k - N_TRAIN) % s == 0:
                        # the index in the predictions/logscores array
                        # Need to add the length of the online updates
                        # and the length of the batch updates
                        # because these models come after those
                        mmm = (
                            len(UPDATE_FREQS_ONLINE)
                            + list(UPDATE_FREQS_BATCH).index(s)
                            + len(UPDATE_FREQS_BATCH)
                        )
                        IDX_PRED = np.arange(k, min(k + s, N_SHORT))

                        pred = estimator[mm].predict_distribution_parameters(
                            X=df_X_batch_scaled.values[IDX_PRED, :],
                            # X=X_numpy[IDX_PRED, :],
                        )
                        pred_scipy = estimator[mm].distribution.theta_to_scipy(pred)
                        diag_s = np.zeros((len(IDX_PRED), H, H))
                        diag_s[:, range(H), range(H)] = np.expand_dims(
                            scaling_s_batch.values, 0
                        )
                        mat_l = np.expand_dims(scaling_m_batch.values, 0)
                        mat_s = np.expand_dims(scaling_s_batch.values, 0)
                        predictions_loc[IDX_PRED - N_TRAIN, mmm, ...] = (
                            pred_scipy["loc"] * mat_s + mat_l
                        )
                        predictions_cov[IDX_PRED - N_TRAIN, mmm, ...] = (
                            diag_s @ pred_scipy["shape"] @ diag_s.swapaxes(-1, -2)
                        )
                        predictions_dof[IDX_PRED - N_TRAIN, mmm, ...] = pred_scipy["df"]
                        timings[IDX_PRED - N_TRAIN, mmm] = train_time / s

                        for t in IDX_PRED - N_TRAIN:
                            logscores[t, mmm] = -st.multivariate_t(
                                loc=predictions_loc[t, mmm, :],
                                shape=predictions_cov[t, mmm, :, :],
                                df=predictions_dof[t, mmm, 0],
                            ).logpdf(prices_test[t, :])

            except Exception as e:
                print(f"Error in model with step {step}: {e}, k={k}, i={i}")


# %% save the results
np.savez(
    os.path.join(FOLDER_RESULTS, "efficient_frontier_comp_time_short.npz"),
    timings=timings,
    logscores=logscores,
    predictions_loc=predictions_loc,
    predictions_cov=predictions_cov,
    predictions_dof=predictions_dof,
    update_freqs_batch=UPDATE_FREQS_BATCH,
    update_freqs_online=UPDATE_FREQS_ONLINE,
    update_freqs=UPDATE_FREQS,
)
# %%
# Load the results
# file = np.load(os.path.join(FOLDER_RESULTS, "efficient_frontier_comp_time_short.npz"))
# timings = file["timings"]
# logscores = file["logscores"]
# predictions_loc = file["predictions_loc"]
# predictions_cov = file["predictions_cov"]
# predictions_dof = file["predictions_dof"]

# UPDATE_FREQS = file["update_freqs"]
# UPDATE_FREQS_BATCH = file["update_freqs_batch"]
# UPDATE_FREQS_ONLINE = file["update_freqs_online"]

# %%
# Plot the results
CMAP = plt.cm.plasma
COLORS = CMAP(np.linspace(0, 1, len(UPDATE_FREQS_ONLINE)))

MO = len(UPDATE_FREQS_ONLINE)
MB = len(UPDATE_FREQS_BATCH)

base_model_train_time = timings[0, 0]

plt.figure(figsize=(7, 4))
# Initial fit training time
plt.axvline(
    x=base_model_train_time,
    color="red",
    alpha=0.75,
    linestyle=":",
    lw=1.5,
    label="Initial fit training time",
)
# Scatter plot for timings and log-scores
# Online
plt.scatter(
    timings[:, :MO].sum(0),
    logscores[:, :MO].mean(0),
    color=COLORS,
    zorder=2,
)
# Batch Increasing Window
plt.scatter(
    timings[:, MO : (MO + MB)].sum(0),
    logscores[:, MO : (MO + MB)].mean(0),
    color=COLORS[(len(UPDATE_FREQS_ONLINE) - len(UPDATE_FREQS_BATCH)) :],
    marker="s",
    zorder=2,
)
plt.scatter(
    timings[:, (MO + MB) :].sum(0),
    logscores[:, (MO + MB) :].mean(0),
    color=COLORS[(len(UPDATE_FREQS_ONLINE) - len(UPDATE_FREQS_BATCH)) :],
    marker="d",
    zorder=2,
)


# Plot the efficient frontier lines
plt.plot(
    timings[:, :MO].sum(0),
    logscores[:, :MO].mean(0),
    color=COLORS[0],
    linestyle="--",
    zorder=1,
)
plt.plot(
    timings[:, MO : (MO + MB)].sum(0),
    logscores[:, MO : (MO + MB)].mean(0),
    color=COLORS[-1],
    linestyle="--",
    zorder=1,
)
plt.plot(
    timings[:, (MO + MB) :].sum(0),
    logscores[:, (MO + MB) :].mean(0),
    color=COLORS[len(COLORS) // 2],
    linestyle="--",
    zorder=1,
)

for i, freq in enumerate(UPDATE_FREQS):
    plt.text(
        timings.sum(0)[i] + 10 if i >= MO else timings.sum(0)[i] - 10,
        logscores.mean(0)[i] + 0.05 if i >= MO else logscores.mean(0)[i] - 0.05,
        str(freq),
        fontsize=9,
        ha="left" if i >= MO else "right",
        va="bottom" if i >= MO else "top",
    )

sm = mpl.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=1, vmax=365))
ax = plt.gca()
cbar = plt.colorbar(sm, orientation="vertical", pad=0.02, ax=ax)
cbar.set_label("Update Frequency (days)")
cbar.set_ticks(np.linspace(1, 365, len(UPDATE_FREQS_ONLINE)))
cbar.set_ticklabels(UPDATE_FREQS_ONLINE)

plt.ylabel("Average Log-Score")
plt.xlabel("Computation Time (seconds, log-scale)")
plt.xscale("log", base=2)
plt.ylim(54.9, 57.1)
plt.xlim(-1000)
plt.grid(which="both", linestyle=":")

labels = ["Online Learning", "Batch (Expanding window)", "Batch (Rolling window)"]
colors = [COLORS[0], COLORS[-1], COLORS[len(COLORS) // 2]]
markers = ["o", "s", "d"]

legend_elements = [
    Line2D(
        [0],
        [0],
        marker=m,
        color=c,
        label=la,
        markeredgecolor="gainsboro",
        markerfacecolor="gainsboro",
        markersize=8,
    )
    for la, c, m in zip(labels, colors, markers)
]

plt.legend(
    handles=legend_elements + plt.gca().get_legend_handles_labels()[0],
    ncol=1,
    loc="upper right",
    # bbox_to_anchor=(0.5, -0.15),
)
plt.title("Efficient Frontier: Computation Time vs. Log-Score")

# Add a zoomed inset for the x-range of the online models
# ax_main = plt.gca()
# axins = inset_axes(ax_main, width="60%", height="40%", loc="upper right", borderpad=2)

# # Plot the same scatter points in the inset
# axins.scatter(
#     timings[:, :MO].sum(0),
#     logscores[:, :MO].mean(0),
#     color=COLORS,
#     zorder=2,
# )
# # Plot the efficient frontier lines in the inset
# axins.plot(
#     timings[:, :MO].sum(0),
#     logscores[:, :MO].mean(0),
#     color=COLORS[0],
#     linestyle="--",
#     zorder=1,
# )
# axins.set_xlim(125, 275)
# axins.set_ylim(55, 56.5)
# axins.grid(which="both", linestyle=":")
# axins.set_xticks(np.arange(125, 276, 25))
# axins.set_yticks(np.arange(55, 56.51, 0.5))
# axins.set_title("Zoom: Online Efficient Frontier", fontsize=10)
# axins.set_facecolor("gainsboro")

# for i, freq in enumerate(UPDATE_FREQS_ONLINE):
#     axins.text(
#         timings.sum(0)[i] - 20,
#         logscores.mean(0)[i] - 0.25 * ,
#         str(freq),
#         fontsize=9,
#         ha="left",
#         va="bottom",
#     )

# mark_inset(
#     ax_main,
#     axins,
#     loc1=2,
#     loc2=4,
#     ec="gainsboro",
#     fill=True,
#     color="gainsboro",
#     zorder=-2
# )


plt.tight_layout()
plt.savefig(
    os.path.join(FOLDER_FIGURES, "eff_frontier.png"),
    **PLT_SAVE_OPTIONS,
)
plt.savefig(
    os.path.join(FOLDER_FIGURES, "eff_frontier.pdf"),
    **PLT_SAVE_OPTIONS,
)
plt.show(block=False)


# %%
# Figure with the differences between batch and online over time
plt.figure(figsize=(7, 4))
for i, f in enumerate(UPDATE_FREQS_BATCH):
    diff = logscores[:, UPDATE_FREQS == f][:, 0] - logscores[:, UPDATE_FREQS == f][:, 1]
    plt.plot(
        diff,
        label=f"Update {f} days: "
        + r"$\text{LS}_\text{o} - \text{LS}_\text{b}$ = "
        + f"{diff.mean():.3f}",
        color=COLORS[UPDATE_FREQS_ONLINE == f],
    )
plt.title("Log-Score Difference Between Batch and Online Updates")
plt.xlim(0, N_TEST_SHORT)
plt.ylim(-250, 250)
plt.yticks(np.linspace(-250, 250, 11))
plt.xticks(np.arange(0, N_TEST_SHORT, 365), dates[N_TRAIN:N_SHORT][::365])
plt.axhspan(
    ymin=0, ymax=plt.ylim()[1], xmin=0, xmax=1, color="red", alpha=0.15, zorder=0
)
plt.axhspan(
    ymin=plt.ylim()[0], ymax=0, xmin=0, xmax=1, color="blue", alpha=0.15, zorder=0
)
plt.grid(ls=":")
plt.text(
    0.98,
    0.98,
    "Batch better",
    color="red",
    fontsize=12,
    ha="right",
    va="top",
    transform=plt.gca().transAxes,
)
plt.text(
    0.98,
    0.02,
    "Online better",
    color="blue",
    fontsize=12,
    ha="right",
    va="bottom",
    transform=plt.gca().transAxes,
)
plt.legend(
    loc="upper left",
    ncol=1,
    #    bbox_to_anchor=(0.5, 1.30),
)
plt.tight_layout()
plt.savefig(
    os.path.join(FOLDER_FIGURES, "batch_vs_online_diff.png"),
    **PLT_SAVE_OPTIONS,
)
plt.savefig(
    os.path.join(FOLDER_FIGURES, "batch_vs_online_diff.pdf"),
    **PLT_SAVE_OPTIONS,
)
plt.show(block=False)


# %%
# Cumulative improvements compared to the best and worst online update
ref_best_online = 0
ref_worst_online = len(UPDATE_FREQS_ONLINE) - 1


print((1 - logscores.mean(0) / logscores.mean(0)[ref_best_online]) * 100)
print((1 - timings.sum(0) / timings.sum(0)[ref_best_online]) * 100)
# %%/ * 100)

print((1 - logscores.mean(0) / logscores.mean(0)[ref_worst_online]) * 100)
print((1 - timings.sum(0) / timings.sum(0)[ref_worst_online]) * 100)


# %%
cumulated_timings = np.cumsum(timings, axis=0)
fig, ax = plt.subplots(1, 2, figsize=(7, 4), sharey=True, sharex=True)
# plt.suptitle("Cumulative Computation Time for Different Update Frequencies")
for i, freq in enumerate(UPDATE_FREQS_ONLINE):
    ax[0].plot(
        cumulated_timings[:, i],
        color=COLORS[i],
        label=f"Update every {freq} days",
        zorder=1,
    )
for i, freq in enumerate(UPDATE_FREQS_BATCH):
    ax[1].plot(
        cumulated_timings[:, MO + i],
        color=COLORS[(len(UPDATE_FREQS_ONLINE) - len(UPDATE_FREQS_BATCH)) + i],
        label=f"Refit every {freq} days",
        zorder=1,
    )
ax[0].set_ylabel("Cumulative Computation Time")
ax[0].set_title("Online Update")
ax[1].set_title("Repeated Batch Fit")

for i in range(2):
    ax[i].set_xlabel("Time Steps")
    ax[i].legend(
        loc="upper left",
    )
    # ax[i].set_yscale("log", base=10)
    ax[i].grid(ls=":")
    ax[i].set_ylim(0, 60 * 60 * 6)
    ax[i].set_yticks(np.arange(0, 60 * 60 * 6 + 1, 60 * 60))
    ax[i].set_yticklabels(
        [f"{int(t/3600)}h" for t in np.arange(0, 60 * 60 * 6 + 1, 60 * 60)]
    )


plt.tight_layout()
plt.savefig(
    os.path.join(FOLDER_FIGURES, "cumulative_computation_time.png"),
    **PLT_SAVE_OPTIONS,
)
plt.savefig(
    os.path.join(FOLDER_FIGURES, "cumulative_computation_time.pdf"),
    **PLT_SAVE_OPTIONS,
)
plt.show(block=False)

# %%
