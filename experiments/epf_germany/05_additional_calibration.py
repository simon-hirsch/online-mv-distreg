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
from joblib import Parallel, delayed
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
# Prediction Bands
# Calculate the prediction bands as in the paper by
# Helmut Lütkepohl, Anna Staszewska-Bystrova, Peter Winker
# https://www.sciencedirect.com/science/article/abs/pii/S0169207013001398

ALPHA = np.array([0.25, 0.2, 0.15, 0.1, 0.05])
prediction_band = np.zeros((N_TEST, N_MODELS, 2, H, len(ALPHA)))


def compute_prediction_band(t, m, a):
    return (
        t,
        m,
        a,
        joint_prediction_band(simulations=sims[t, m], alpha=ALPHA[a], power=2),
    )


results = Parallel(n_jobs=4)(
    delayed(compute_prediction_band)(t, m, a)
    for t, m, a in product(range(N_TEST), range(N_MODELS), range(len(ALPHA)))
)

# %%
for t, m, a, band in results:
    prediction_band[t, m, :, :, a] = band


# %% Calculate the prediction band coverage
cover_upper = (prediction_band[:, :, 1, :] >= prices_test[:, None, :, None]).all(-2)
cover_lower = (prediction_band[:, :, 0, :] <= prices_test[:, None, :, None]).all(-2)
prediction_band_cover = np.logical_and(cover_upper, cover_lower)
prediction_band_width = prediction_band[:, :, 1] - prediction_band[:, :, 0]

# %%
miscoverage = prediction_band_cover.mean(0) - (1 - ALPHA)
width_median = np.median(prediction_band_width, (0, 2))
width_mean = np.mean(prediction_band_width, (0, 2))

# %%
plt.figure(figsize=(8, 4))
plt.gca().set_prop_cycle(plt.cycler(color=plt.cm.turbo(np.linspace(0, 1, N_MODELS))))
for m, model in enumerate(MODEL_NAMES):
    if not "ar_cp" in model:
        plt.plot(
            1 - ALPHA,
            miscoverage[m, :],
            ls=LS[m],
            label=MODEL_NAMES_NICE[m],
            marker="o",
        )

plt.xticks(1 - ALPHA)
plt.axhline(0, color="red")
plt.ylim(-0.2, 0.2)
plt.grid(ls=":")
plt.xlabel("Nominal Prediction Band Coverage Probability")
plt.tight_layout()
plt.ylabel("Miscoverage")
plt.title("Prediction Band Miscoverage by Model")
plt.legend(ncol=3, fontsize=10, loc="lower center", bbox_to_anchor=(0.5, -0.5))


# %%
plt.figure(figsize=(8, 4))
plt.gca().set_prop_cycle(plt.cycler(color=plt.cm.turbo(np.linspace(0, 1, N_MODELS))))
for m, model in enumerate(MODEL_NAMES):
    if not "ar_cp" in model:
        plt.plot(
            1 - ALPHA,
            width_mean[m, :],
            ls=LS[m],
            label=MODEL_NAMES_NICE[m],
            marker="o",
        )
plt.xticks(1 - ALPHA)
plt.grid(ls=":")
plt.ylabel("Average Prediction Band Width")
plt.xlabel("Nominal Prediction Band Coverage Probability")
plt.tight_layout()
plt.title("Mean Prediction Band Width by Model")
plt.legend(ncol=3, fontsize=10, loc="lower center", bbox_to_anchor=(0.5, -0.5))
# %%


# %%
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].set_prop_cycle(plt.cycler(color=plt.cm.turbo(np.linspace(0, 1, N_MODELS))))
for m, model in enumerate(MODEL_NAMES):
    if model not in "ar_cp":
        axes[0].plot(
            1 - ALPHA,
            miscoverage[m, :],
            ls=LS[m],
            label=MODEL_NAMES_NICE[m],
            marker="o",
        )

axes[0].set_xticks(1 - ALPHA)
axes[0].axhline(0, color="red")
axes[0].set_ylim(-0.2, 0.2)
axes[0].grid(ls=":")
axes[0].set_xlabel("Nominal Prediction Band Coverage Probability")
axes[0].set_ylabel("Miscoverage")
axes[0].set_title("Prediction Band Miscoverage by Model")

axes[1].set_prop_cycle(plt.cycler(color=plt.cm.turbo(np.linspace(0, 1, N_MODELS))))
for m, model in enumerate(MODEL_NAMES):
    if model not in "ar_cp":
        axes[1].plot(
            1 - ALPHA,
            width_mean[m, :],
            ls=LS[m],
            label=MODEL_NAMES_NICE[m],
            marker="o",
        )

axes[1].set_xticks(1 - ALPHA)
axes[1].grid(ls=":")
axes[1].set_ylabel("Average Prediction Band Width")
axes[1].set_xlabel("Nominal Prediction Band Coverage Probability")
axes[1].set_title("Mean Prediction Band Width by Model")

handles, labels = axes[1].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    ncol=3,
    fontsize=10,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.15),
)
plt.tight_layout()
plt.subplots_adjust(bottom=0.3)
plt.savefig(
    os.path.join(FOLDER_FIGURES, "calibration_prediction_bands.png"), **PLT_SAVE_OPTIONS
)
plt.savefig(
    os.path.join(FOLDER_FIGURES, "calibration_prediction_bands.pdf"), **PLT_SAVE_OPTIONS
)
# %%
np.savez(
    os.path.join(FOLDER_RESULTS, "prediction_band_calibration.npz"),
    prediction_band=prediction_band,
    miscoverage=miscoverage,
    width_mean=width_mean,
    width_median=width_median,
    model_names=MODEL_NAMES,
)
# %%
