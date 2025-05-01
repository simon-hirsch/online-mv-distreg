import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns

df_X = pd.read_csv("prepared_x.csv", index_col=0)
df_y = pd.read_csv("prepared_y.csv", index_col=0, parse_dates=True)


IDX_TRAIN = df_X.flag.to_numpy() == "train"
IDX_TEST = df_X.flag.to_numpy() == "test"

N_TRAIN = np.sum(IDX_TRAIN)
N_TEST = np.sum(IDX_TEST)
H = 24
N = df_X.shape[0]

IDX_EVAL = np.arange(182, N_TEST)


data = df_y[:N_TRAIN]
data.columns = range(24)

N_DAYS = 180
DAY_START = 724
DAY_END = DAY_START + N_DAYS

START_DATE = data.index[DAY_START]
END_DATE = data.index[DAY_END]


## Figure with time series from the power prices
print(START_DATE)


plt.figure(figsize=(10, 10))
plt.ylim(-100, 225)
plt.xlim(0, 23)
data.iloc[DAY_START:DAY_END, :].T.plot(ax=plt.gca(), cmap="turbo", marker="o")
handles, labels = plt.gca().get_legend_handles_labels()
labels = data.index[DAY_START:DAY_END].date
plt.grid()
plt.ylabel("Price [EUR/MWh]")
plt.xlabel("Delivery Hour")
plt.xticks(np.arange(24), rotation=90)
plt.legend(handles[::10], labels[::10], ncol=3)
plt.tight_layout()
plt.savefig("figures/price_time_series_by_hour.png")
# plt.show()


plt.figure(figsize=(20, 10), dpi=300)
df_y.plot(cmap="turbo", legend=None, ax=plt.gca())
plt.ylim(-100, 225)
plt.xlim(df_y.index.min(), df_y.index.max())
plt.axvline(df_y.index[N_TRAIN], color="black", ls=":")
plt.grid()
plt.axvspan(START_DATE, END_DATE, color="grey", alpha=0.25)
plt.ylabel("Price [EUR/MWh]")
plt.xlabel("Date")
plt.legend(np.arange(24), ncol=4, title="Delivery Hour")
plt.tight_layout()
plt.savefig("figures/price_time_series.png")
# plt.show()


## Get benchmarks for the correlation plot of the residuals
residuals = np.load("results/pred_univariate_benchmark.npz")["residuals"]

corr_prices = np.corrcoef(df_y[:N_TRAIN], rowvar=False)
corr_residu = np.corrcoef(residuals[:N_TRAIN], rowvar=False)

corrmat = np.full((24, 24), np.nan)
corrmat[np.tril_indices(24, k=-1)] = corr_prices[np.tril_indices(24, k=-1)]
corrmat[np.triu_indices(24, k=1)] = corr_residu[np.triu_indices(24, k=1)]


## Get the standard error for the correlation matrix and the p-value
corrmat_standard_error = np.sqrt((1 - corrmat**2) / (N_TRAIN - 2))

sns.heatmap(
    st.t(df=N_TRAIN - 2).cdf(1 - corrmat / corrmat_standard_error), vmin=0, vmax=0.1
)

## Make the real heatmap
plt.figure(figsize=(9, 7), dpi=300)
sns.heatmap(
    corrmat,
    vmin=0,
    vmax=1,
    cmap="turbo_r",
    cbar_kws={"label": r"Pearson Correlation $\rho$"},
    square=True,
    annot=True,
    annot_kws={"size": 7},
)
# plt.title("Correlation matrix \n Lower triangle shows unconditional correlation \n Upper triangle shows conditional on mean prediction")
plt.ylabel("Delivery Hour")
plt.xlabel("Delivery Hour")
plt.tight_layout()
plt.savefig("figures/correlation.png")
# plt.show()


# Plot results from forecasting study
simulations = np.load("results/sims_multivariate.npz")["simulations"]

t = 100
m0 = 5
m1 = 4
idx_sims = np.arange(100)


fig, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 5), dpi=300)
ax[0].set_prop_cycle(color=plt.cm.turbo(np.linspace(0, 1, len(idx_sims))))
ax[0].plot(simulations[t, m0, idx_sims].T)
ax[0].plot(
    simulations[t, m0, :].T.mean(1), color="black", ls="--", label="Mean Prediction"
)
ax[0].set_title("Model using LRA-based scale matrix.")
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
        df_y[IDX_TEST].values[t],
        color="black",
        label="Realized Spot Price",
        lw=2,
        ds="steps",
    )
    ax[i].legend()

plt.tight_layout()
plt.savefig("figures/simulations_comp.png", dpi=300)
# plt.show()


## Figure for the result plots
pred_scale = np.load("results/pred_multivariate.npz")["predictions_cov"]
pred_dof = np.load("results/pred_multivariate.npz")["predictions_dof"]

m = 4
pred_dof[:, m, 0]
pred_cov = pred_scale[:, m] * np.expand_dims(
    pred_dof[:, m, 0] / (pred_dof[:, m, 0] - 2), (1, 2)
)
pred_corr = pred_cov / (
    pred_cov[:, range(H), range(H), None] ** 0.5
    @ pred_cov[:, None, range(H), range(H)] ** 0.5
)

n_plots = 7
start = 107

fig, axes = plt.subplots(1, n_plots, figsize=(20, 6), sharey=True)

for i in range(n_plots):
    if i == 0:
        axes[i].set_ylabel("Delivery Hour")
    im = axes[i].imshow(pred_cov[start + i], cmap="turbo")
    axes[i].set_title(f"{df_X[IDX_TEST].index[start+i]}")
    axes[i].set_xticks(np.arange(24, step=2))
    axes[i].set_xticklabels(np.arange(24, step=2), rotation=90)
    axes[i].set_yticks(np.arange(24, step=2))
    axes[i].set_yticklabels(np.arange(24, step=2), rotation=0)
    axes[i].set_xlabel("Delivery Hour")

plt.tight_layout()
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.4)
plt.savefig(
    "figures/illustrative_covariance_matrices.png",
    dpi=300,
    bbox_inches="tight",
)
# plt.show()

# Same for the IWSM short paper

plt.rcParams.update({"text.usetex": True, "font.family": "CM"})

n_plots = 4
start = 112

fig, axes = plt.subplots(1, n_plots, figsize=(20, 6), sharey=True)

for i in range(n_plots):
    if i == 0:
        axes[i].set_ylabel("Delivery Hour")
    im = axes[i].imshow(pred_cov[start + i], cmap="turbo")
    axes[i].set_title(f"{df_X[IDX_TEST].index[start+i]}")
    axes[i].set_xticks(np.arange(24, step=2))
    axes[i].set_xticklabels(np.arange(24, step=2), rotation=90)
    axes[i].set_yticks(np.arange(24, step=2))
    axes[i].set_yticklabels(np.arange(24, step=2), rotation=0)
    axes[i].set_xlabel("Delivery Hour")

plt.tight_layout()
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.4)
plt.savefig(
    "figures/illustrative_covariance_matrices_iwsm.png",
    dpi=300,
    bbox_inches="tight",
    backend="pgf",
)
