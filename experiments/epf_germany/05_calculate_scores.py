# %%
# Allow running as notebook and as script
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# %%
# It takes roughly 60 mins to calculate the scores
import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import scoringrules as sr
from const_and_helper import (
    FOLDER_DATA,
    FOLDER_FIGURES,
    FOLDER_RESULTS,
    MODEL_NAMES_MAPPING,
    PLT_SAVE_OPTIONS,
    PLT_TEX_OPTIONS,
)
from const_and_helper.distribution import gaussian_copula_log_likelihood
from const_and_helper.evaluation import (  # noqa: E501
    dawid_sebastiani_scoore,
    energy_score_akr,
    energy_score_fast,
    joint_prediction_band,
)
from ondil.distributions import StudentT
from tqdm import tqdm

plt.rcParams.update(PLT_TEX_OPTIONS)


# %%
# Load data
# Prepare the untransformed prices
df_X = pd.read_csv(os.path.join(FOLDER_DATA, "prepared_long_x.csv"), index_col=0)
df_y = pd.read_csv(os.path.join(FOLDER_DATA, "prepared_long_y.csv"), index_col=0)

KEY = "simulations"
IDX_TRAIN = df_X.flag.to_numpy() == "train"
IDX_TEST = df_X.flag.to_numpy() == "test"
N_TRAIN = np.sum(IDX_TRAIN)
N_TEST = np.sum(IDX_TEST)
H = df_y.shape[1]
N = df_X.shape[0]

prices = pd.read_csv(os.path.join(FOLDER_DATA, "prices_untransformed.csv"), index_col=0)
prices_numpy = prices.values.astype(np.float32)
prices_test = prices_numpy[IDX_TEST, :]

# %%
# Load the simulations

FILE_MV = os.path.join(FOLDER_RESULTS, "sims_multivariate.npz")
FILE_COPULA = os.path.join(FOLDER_RESULTS, "sims_copula.npz")
FILE_AR = os.path.join(FOLDER_RESULTS, "sims_univariate_benchmark.npz")
FILE_CP = os.path.join(FOLDER_RESULTS, "benchmark_cp.npz")
FILE_GARCH = os.path.join(FOLDER_RESULTS, "sims_garch_benchmark.npz")

sims = np.concatenate(
    (
        np.load(FILE_AR)[KEY].astype(np.float32),
        np.load(FILE_CP)[KEY].astype(np.float32),
        np.load(FILE_GARCH)[KEY].astype(np.float32),
        np.load(FILE_COPULA)[KEY].astype(np.float32),
        np.load(FILE_MV)[KEY].astype(np.float32),
    ),
    axis=1,
)


N_MODELS = sims.shape[1]
N_SIMS = sims.shape[-2]

# %%
MODEL_NAMES = np.concatenate(
    (
        np.load(FILE_AR)["model_names"],
        np.load(FILE_CP)["model_names"],
        np.load(FILE_GARCH)["model_names"],
        np.load(FILE_COPULA)["model_names"],
        np.load(FILE_MV)["model_names"],
    )
)
MODEL_NAMES_NICE = [MODEL_NAMES_MAPPING.get(name) for name in MODEL_NAMES]

LS_MAPPING = {"LARX": ":", "oDis": "--", "oMvD": "-"}

LS = [LS_MAPPING[i[:4]] for i in MODEL_NAMES_NICE]


# %%
# Root Mean Squared Error, MAE, CRPS
error_mean = np.expand_dims(prices_test, 1) - sims.mean(-2)
error_med = np.expand_dims(prices_test, 1) - np.median(sims, axis=-2)
error_crps = np.zeros_like(error_mean)
for m in range(N_MODELS):
    error_crps[:, m, :] = sr.crps_ensemble(
        prices_test,
        sims[:, m, ...],
        m_axis=-2,
    )

# %% - Add the conformal prediction model since the sims are all nan
error_mean[:, 2] = np.load(FILE_CP)["error_mean"]
error_med[:, 2] = np.load(FILE_CP)["error_med"]
error_crps[:, 2, :] = np.load(FILE_CP)["error_crps"]

# %%
hour_rmse = np.mean(error_mean**2, 0) ** 0.5
hour_mae = np.mean(np.abs(error_med), 0)
hour_crps = np.mean(error_crps, 0)


# %%
# Prediction Bands
# Calculate the prediction bands as in the paper by
# Helmut Lütkepohl, Anna Staszewska-Bystrova, Peter Winker
# https://www.sciencedirect.com/science/article/abs/pii/S0169207013001398

prediction_band = np.zeros((N_TEST, N_MODELS, 2, H))
for t, m in product(range(N_TEST), range(N_MODELS)):
    prediction_band[t, m] = joint_prediction_band(
        simulations=sims[t, m], alpha=0.05, power=2
    )


# %% Add conformal prediction bands
prediction_band[:, 2, :, :] = np.load(FILE_CP)["jpb"].swapaxes(-2, -1)

# %% Calculate the prediction band coverage
cover_upper = (prediction_band[:, :, 1] >= prices_test[:, None, :]).all(-1)
cover_lower = (prediction_band[:, :, 0] <= prices_test[:, None, :]).all(-1)
prediction_band_cover = np.logical_and(cover_upper, cover_lower)
prediction_band_width = prediction_band[:, :, 1] - prediction_band[:, :, 0]

# %%
miscoverage = prediction_band_cover.mean(0) - (1 - 0.05)
median_width = np.median(prediction_band_width, (0, 2))

# TODO: Add the model names to the plot
# Maybe take a nice color scheme
# Or make this two plots?
lower = -0.40
upper = -lower
ylim = 150

plt.figure(figsize=(8, 4))
plt.title("Joint Prediction Coverage and Band Width")
ax1 = plt.gca()
ax1.scatter(np.arange(N_MODELS), miscoverage, color="red", s=100)
ax1.set_ylim(lower, upper)
ax1.set_yticks(np.linspace(lower, upper, 11))
ax1.set_ylabel("Miscoverage (Dotted)")
ax1.axhline(y=0, color="red", ls=":")

ax2 = ax1.twinx()
ax2.bar(np.arange(N_MODELS), median_width, color="blue", alpha=0.5)
ax2.set_ylabel("Prediction Band Width (Bars)")
ax2.set_ylim(0, ylim)
ax2.set_yticks(np.linspace(0, ylim, 11))
ax2.grid(ls=":", axis="y")

plt.tight_layout()
plt.savefig(os.path.join(FOLDER_FIGURES, "prediction_bands.pdf"), **PLT_SAVE_OPTIONS)
plt.show(block=False)
# Generally seems to be too wide
# But the width of the prediction bands decreases quite a bit MCD distribution
# This is actually quite a nice result

# %%
m = 0
t = 0
a = 0.5
gen = np.random.default_rng(123)
SIM_SUBSET = gen.choice(np.arange(N_SIMS), 1000, replace=False)

for m in range(2):
    example_qpb = np.vstack(
        (
            np.quantile(sims[t, m, SIM_SUBSET], 0.25, 0),
            np.quantile(sims[t, m, SIM_SUBSET], 0.75, 0),
        )
    ).T
    example_jpb = joint_prediction_band(
        simulations=sims[t, m, SIM_SUBSET], alpha=0.5, power=2
    )

    width_qpb = example_qpb[:, 1] - example_qpb[:, 0]
    width_jpb = example_jpb[1] - example_jpb[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150, sharey=True)
    for i in range(2):
        axes[i].plot(sims[t, m, SIM_SUBSET].T, color="grey", alpha=0.1)
    axes[0].set_ylabel("Price")
    axes[0].fill_between(
        np.arange(H),
        example_qpb[:, 0],
        example_qpb[:, 1],
        color="yellow",
        alpha=0.5,
        label="QPB (alpha=0.5)",
        edgecolor="black",
        zorder=2,
    )
    axes[1].fill_between(
        np.arange(H),
        example_jpb[0],
        example_jpb[1],
        color="lightblue",
        alpha=0.5,
        label="JPB (alpha=0.5)",
        edgecolor="black",
        zorder=2,
    )
    for i in range(2):
        axes[i].set_xlabel("Hour")
        axes[i].legend(ncol=2, loc="upper left")
        axes[i].set_xticks(np.arange(H, step=2))
        axes[i].grid(ls=":")
    plt.suptitle(
        f"Quantile Prediction Bands and Joint Prediction Bands for Model {MODEL_NAMES_NICE[m]}"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(FOLDER_FIGURES, f"prediction_bands_example_{m}.pdf"),
        **PLT_SAVE_OPTIONS,
    )
    plt.savefig(
        os.path.join(FOLDER_FIGURES, f"prediction_bands_example_{m}.png"),
        **PLT_SAVE_OPTIONS,
    )
    plt.show(block=False)


# %%
# Calculate the aggregated scores for the models
vs05 = np.full((N_TEST, N_MODELS), np.nan)
vs10 = np.full((N_TEST, N_MODELS), np.nan)
es = np.full((N_TEST, N_MODELS), np.nan)
dss = np.full((N_TEST, N_MODELS), np.nan)

# The calculation of the energy score
# is quite slow, so we use tqdm to show the progress
# We iterate over the test set and the models
# since the scoring rules are quite memory intensive
# so we cannot calculate them for all models at once

print("Calculating Energy and Variogram Scores")

for t, m in tqdm(product(range(N_TEST), range(N_MODELS))):
    try:
        vs05[t, m] = sr.variogram_score(prices_test[t], sims[t, m, :, :], p=0.5)
        vs10[t, m] = sr.variogram_score(prices_test[t], sims[t, m, :, :], p=1)
        es[t, m] = energy_score_fast(prices_test[t], sims[t, m, :, :])
        dss[t, m] = dawid_sebastiani_scoore(prices_test[t], sims[t, m, ...])
    except Exception as _:
        pass
        # print(f"Could not calculate score for model {m}.")


# %%
# Calculation of the log-likelihood scores
# We calculate the scores by model as this is different per model type
# Note that we cannot calculate the scores for all models at all (conformal)

# %% Univariate Log-Scores
ls_univariate = np.zeros((N_TEST, 2))
file_univariate = np.load(
    file="experiments/epf_germany/results_revision/pred_univariate_benchmark.npz"
)
predictions_loc_ar = file_univariate["predictions_loc"]
predictions_cov_ar = file_univariate["predictions_cov"]
for t, m in product(range(N_TEST), range(2)):
    ls_univariate[t, m] = -st.multivariate_normal(
        mean=predictions_loc_ar[t],
        cov=predictions_cov_ar[t] if m == 1 else np.diag(predictions_cov_ar[t]),
    ).logpdf(prices_test[t, :])

# %% Copula Log-Scores

ls_copula = np.zeros((N_TEST, 3))
file_copula = np.load(file="experiments/epf_germany/results_revision/pred_copula.npz")
file_univariate_distreg = np.load(
    "experiments/epf_germany/results_revision/univariate_predictions_distreg.npz"
)

pred_uni_copula = file_copula["predictions_uni"]
pred_cov_copula = file_copula["predictions_cov"]

marginal_loglikelihood = np.zeros((N_TEST, H))
marginal_predictions = file_univariate_distreg["predictions_outofsample"]

for h in range(H):
    marginal_loglikelihood[:, h] = StudentT().pdf(
        prices_test[:, h], marginal_predictions[:, h, :]
    )

ls_copula[:, 0] = -np.log(marginal_loglikelihood).sum(1)

for c, m in enumerate(range(1, 3)):
    ls_copula[:, m] = -(
        gaussian_copula_log_likelihood(pred_uni_copula, pred_cov_copula[:, c])
        + np.log(marginal_loglikelihood).sum(1)
    )


# %% Multivariate Log-Scores
ls_multivariate = np.zeros((N_TEST, 9))
file_multivariate = np.load(
    file="experiments/epf_germany/results_revision/pred_multivariate.npz"
)
predictions_loc_mv = file_multivariate["predictions_loc"]
predictions_cov_mv = file_multivariate["predictions_cov"]
predictions_dof_mv = file_multivariate["predictions_dof"]

for t, m in product(range(N_TEST), range(9)):
    try:
        ls_multivariate[t, m] = -st.multivariate_t(
            loc=predictions_loc_mv[t, m],
            shape=predictions_cov_mv[t, m],
            df=predictions_dof_mv[t, m],
        ).logpdf(prices_test[t, :])
    except Exception as _:
        pass

ls_garch = np.zeros((N_TEST, 1))
file_garch = np.load(
    file="experiments/epf_germany/results_revision/pred_garch_benchmark.npz"
)
predictions_loc_garch = file_garch["predictions_loc"]
predictions_std_garch = file_garch["predictions_std"]
for t in range(N_TEST):
    ls_garch[t, 0] = -st.multivariate_normal(
        mean=predictions_loc_garch[t],
        cov=np.diag(predictions_std_garch[t] ** 2),
    ).logpdf(prices_test[t, :])

# %%
# Add all the log-scores together
ls_cp = np.full((N_TEST, 1), np.nan)
ls = np.hstack(
    (
        ls_univariate,
        ls_cp,
        ls_garch,
        ls_copula,
        ls_multivariate,
    )
)


# %%
# Save all the results
np.savez(
    file="experiments/epf_germany/results_revision/scores.npz",
    ls=ls,
    vs05=vs05,
    vs10=vs10,
    es=es,
    dss=dss,
    prediction_band_cover=prediction_band_cover,
    prediction_band_width=prediction_band_width,
    miscoverage=miscoverage,
    median_width=median_width,
    error_crps=error_crps,
    error_mean=error_mean,
    error_med=error_med,
    model_names=MODEL_NAMES,
)
print("Successfully saved the scores.")
