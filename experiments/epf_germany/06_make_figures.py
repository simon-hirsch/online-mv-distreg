# %%
# Allow running as notebook and as script
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# %%
# Import all packages and data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
from const_and_helper import (
    FOLDER_DATA,
    FOLDER_FIGURES,
    FOLDER_RESULTS,
    FOLDER_TABLES,
    PLT_SAVE_OPTIONS,
    PLT_TEX_OPTIONS,
    H,
)
from matplotlib.gridspec import GridSpec

plt.rcParams.update(PLT_TEX_OPTIONS)

CMAP_CORR_AND_COV_MATS = "magma_r"

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

prices = pd.read_csv(
    os.path.join(FOLDER_DATA, "prices_untransformed.csv"), index_col=0, parse_dates=True
)

IDX_TRAIN = df_X.flag.to_numpy() == "train"
IDX_TEST = df_X.flag.to_numpy() == "test"

N_TRAIN = np.sum(IDX_TRAIN)
N_TEST = np.sum(IDX_TEST)
N = df_X.shape[0]

IDX_EVAL = np.arange(182, N_TEST)

data = prices
data.index = pd.to_datetime(data.index)
data.columns = range(24)

N_DAYS = 180
DAY_START = 724
DAY_END = DAY_START + N_DAYS

START_DATE = data.index[DAY_START].date()
END_DATE = data.index[DAY_END].date()


# %%
mask_pre = prices.index.year < 2021
mask_2021 = prices.index.year == 2021
mask_2022 = prices.index.year == 2022
mask_2032 = prices.index.year == 2023

masks = {
    "Pre-2021": mask_pre,
    "2021": mask_2021,
    "2022": mask_2022,
    "2023": mask_2032,
}

summary = (
    pd.DataFrame(
        {
            label: {
                "Mean $\mu$": prices.loc[mask].values.mean(),
                "Std $\sigma$": prices.loc[mask].values.std(),
                "MAD": st.median_abs_deviation(prices.loc[mask].values.ravel()),
                "Min": prices.loc[mask].values.min(),
                "Max": prices.loc[mask].values.max(),
            }
            for label, mask in masks.items()
        }
    )
    .round(1)
    .T
)

summary.to_latex(
    os.path.join(FOLDER_TABLES, "summary_statistics.tex"),
    float_format="%.1f",
)


# %%
fig = plt.figure(figsize=(12, 5), dpi=300)
gs = GridSpec(1, 2, width_ratios=[4, 2], figure=fig)

legend_step = 30

# First plot (time series by hour)
ax0 = fig.add_subplot(gs[1])
ax0.set_xlim(0, 23)
ax0.set_ylim(-100, 250)
data.iloc[DAY_START:DAY_END, :].T.plot(ax=ax0, cmap="turbo", marker="o")
handles, labels = ax0.get_legend_handles_labels()
labels = data.index[DAY_START:DAY_END].date
ax0.grid()
ax0.set_ylabel("Price [EUR/MWh]")
ax0.set_xlabel("Delivery Hour")
ax0.set_xticks(np.arange(24))
ax0.set_xticklabels(np.arange(24), rotation=90)
ax0.legend(handles[::legend_step], labels[::legend_step], ncol=2, loc="upper center")
ax0.set_title("Power Prices by Hour")
ax0.set_facecolor("gainsboro")

# Second plot (full time series)
ax1 = fig.add_subplot(gs[0])
data.plot(cmap="turbo", legend=None, ax=ax1)
ax1.set_ylim(-200, 1000)
ax1.set_xlim(df_y.index.min(), df_y.index.max())
ax1.axvline(df_y.index[N_TRAIN], color="black", ls=":")
ax1.grid()
ax1.axvspan(START_DATE, END_DATE, color="gainsboro")
ax1.set_ylabel("Price [EUR/MWh]")
ax1.set_xlabel("Date")
ax1.legend(np.arange(24), ncol=6, title="Delivery Hour")
ax1.set_title("Power Prices Over Time")

plt.tight_layout()
plt.savefig(
    "experiments/epf_germany/figures/price_time_series_subfigures.png",
    **PLT_SAVE_OPTIONS,
)
plt.savefig(
    "experiments/epf_germany/figures/price_time_series_subfigures.pdf",
    **PLT_SAVE_OPTIONS,
)
plt.show(block=False)


# %%
## Get benchmarks for the correlation plot of the residuals
residuals = np.load(
    "experiments/epf_germany/results_revision/pred_univariate_benchmark.npz"
)["residuals"]

corr_prices = np.corrcoef(df_y[:N_TRAIN], rowvar=False)
corr_residu = np.corrcoef(residuals[:N_TRAIN], rowvar=False)

corrmat = np.full((24, 24), np.nan)
corrmat[np.tril_indices(24, k=-1)] = corr_prices[np.tril_indices(24, k=-1)]
corrmat[np.triu_indices(24, k=1)] = corr_residu[np.triu_indices(24, k=1)]


## Get the standard error for the correlation matrix and the p-value
corrmat_standard_error = np.sqrt((1 - corrmat**2) / (N_TRAIN - 2))

## Make the real heatmap
plt.figure(figsize=(9, 7), dpi=300)
sns.heatmap(
    corrmat,
    vmin=0,
    vmax=1,
    cmap=CMAP_CORR_AND_COV_MATS,
    cbar_kws={"label": r"Pearson Correlation $\rho$"},
    square=True,
    annot=True,
    annot_kws={"size": 7},
)
# plt.title("Correlation matrix \n Lower triangle shows unconditional correlation \n Upper triangle shows conditional on mean prediction")
plt.ylabel("Delivery Hour")
plt.xlabel("Delivery Hour")
plt.tight_layout()
plt.savefig("experiments/epf_germany/figures/correlation.png", **PLT_SAVE_OPTIONS)
plt.savefig("experiments/epf_germany/figures/correlation.pdf", **PLT_SAVE_OPTIONS)
# plt.show(block=False)

# %%
# Plot results from forecasting study
simulations = np.load("experiments/epf_germany/results_revision/sims_multivariate.npz")[
    "simulations"
]

# %%
t = 100
m0 = 3
m1 = 4
idx_sims = np.arange(250)

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 5), dpi=300)
ax[0].set_prop_cycle(color=plt.cm.turbo(np.linspace(0, 1, len(idx_sims))))
ax[0].plot(simulations[t, m0, idx_sims].T)
ax[0].plot(
    simulations[t, m0, :].T.mean(1), color="black", ls="--", label="Mean Prediction"
)
ax[0].set_title("Model assuming independent delivery hours")
ax[0].set_ylabel("Price [EUR/MWh]")

# Plot the second model
ax[1].set_prop_cycle(color=plt.cm.turbo(np.linspace(0, 1, len(idx_sims))))
ax[1].plot(simulations[t, m1, idx_sims].T)
ax[1].plot(
    simulations[t, m1, :].T.mean(1), color="black", ls="--", label="Mean Prediction"
)
ax[1].set_title("Model using CD-based scale matrix")

for i in range(2):
    ax[i].set_xlabel("Delivery Hour")
    ax[i].grid()
    ax[i].set_xlim(0, 23)
    ax[i].set_xticks(np.arange(24))
    ax[i].set_xticklabels(np.arange(24), rotation=90)
    ax[i].plot(
        prices[IDX_TEST].values[t],
        color="black",
        label="Realized Spot Price",
        lw=2,
        ds="steps",
    )
    ax[i].legend()

plt.tight_layout()
plt.savefig(os.path.join(FOLDER_FIGURES, "simulations_comp.png"), **PLT_SAVE_OPTIONS)
plt.savefig(os.path.join(FOLDER_FIGURES, "simulations_comp.pdf"), **PLT_SAVE_OPTIONS)
plt.show(block=False)

# %%
## Estimated scale matrices
file = np.load(os.path.join(FOLDER_RESULTS, "pred_multivariate.npz"))
pred_scale = file["predictions_cov"]
pred_dof = file["predictions_dof"]

m = 1
n_plots = 7
start = 107
fig, axes = plt.subplots(1, n_plots, figsize=(20, 6), sharey=True)

for i in range(n_plots):
    if i == 0:
        axes[i].set_ylabel("Delivery Hour")
    im = axes[i].imshow(pred_scale[start + i, m], cmap=CMAP_CORR_AND_COV_MATS)
    axes[i].set_title(f"{df_X[IDX_TEST].index[start+i]}")
    axes[i].set_xticks(np.arange(24, step=2))
    axes[i].set_xticklabels(np.arange(24, step=2), rotation=90)
    axes[i].set_yticks(np.arange(24, step=2))
    axes[i].set_yticklabels(np.arange(24, step=2), rotation=0)
    axes[i].set_xlabel("Delivery Hour")

plt.tight_layout()
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.4)
plt.savefig(
    os.path.join(FOLDER_FIGURES, "illustrative_covariance_matrices.png"),
    **PLT_SAVE_OPTIONS,
)
plt.savefig(
    os.path.join(FOLDER_FIGURES, "illustrative_covariance_matrices.pdf"),
    **PLT_SAVE_OPTIONS,
)
# plt.show(block=False)


# %%
# Make a figure for the development of the correlation and the standard deviation
# TODO: This is not perfect yet

idx1 = 0
idx2 = 180
step = 30

pred_dof[:, m, 0]
pred_cov = pred_scale[:, m] * np.expand_dims(
    pred_dof[:, m, 0] / (pred_dof[:, m, 0] - 2), (1, 2)
)
pred_corr = pred_cov / (
    pred_cov[:, range(H), range(H), None] ** 0.5
    @ pred_cov[:, None, range(H), range(H)] ** 0.5
)

test_dates = X.index[IDX_TEST].astype(str)

fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True, dpi=300)
# im0 = axes[0].imshow(
#     pred_cov[idx1:idx2, range(H), range(H)].T ** 0.5,
#     aspect="auto",
#     cmap=CMAP_CORR_AND_COV_MATS,
#     vmax=np.max(pred_cov[idx1:idx2, range(H), range(H)].T ** 0.5),
# )
im0 = axes[0].imshow(
    pred_scale[idx1:idx2, m, range(H), range(H)].T,
    aspect="auto",
    cmap=CMAP_CORR_AND_COV_MATS,
    vmin=0,
    vmax=100,
)

# axes[0].set_title(r"Standard Deviation $\hat{\sigma}_{t,h}$")
axes[0].set_title(
    r"Diagonal of the predicted scale matrix $\operatorname{diag}(\hat{\boldsymbol{\Sigma}}_{t})$"
)
axes[0].set_ylabel("Delivery Hour")
axes[0].set_yticks(np.arange(0, H, step=4))

counters = [
    "Zero-th",
    "First",
    "Second",
    "Third",
]

fig.colorbar(
    im0,
    ax=axes[0],
    label="Predicted Scale",
    aspect=5,
    ticks=np.linspace(0, 100, 5),
    location="left",
)
for i in range(1, len(axes)):
    axes[i].set_title(
        f"{counters[i]} off-diagonal of the predicted correlation matrix $\hat{{\\rho}}_{{t,h,h+{i}}}$"
    )
    im = axes[i].imshow(
        pred_corr[idx1:idx2, range(i, H), range(H - i)].T,
        aspect="auto",
        cmap=CMAP_CORR_AND_COV_MATS,
        vmin=0.25,
        vmax=1,
    )
    axes[i].set_ylabel("Hour $h$")
    axes[i].set_yticks(np.arange(0, H - i, step=4))
    fig.colorbar(
        im,
        ax=axes[i],
        label="Correlation",
        aspect=5,
        ticks=np.linspace(0.25, 1, 5),
        location="left",
    )

axes[-1].set_xlabel("Time Index")
axes[-1].set_xticks(np.arange(0, idx2 - idx1, step))
axes[-1].set_xticklabels(test_dates[np.arange(idx1, idx2, step)])

plt.savefig(
    os.path.join(FOLDER_FIGURES, "evoluation_cov_and_corr.pdf"),
    **PLT_SAVE_OPTIONS,
)
plt.savefig(
    os.path.join(FOLDER_FIGURES, "evoluation_cov_and_corr.png"),
    **PLT_SAVE_OPTIONS,
)
plt.tight_layout()
plt.show(block=False)

# %%
