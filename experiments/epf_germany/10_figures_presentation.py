# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
from matplotlib import rcParams

y = pd.read_csv("experiments/epf_germany/prepared_y.csv", index_col=0)

DAY = 1400
H = 24

# %%
y.iloc[:10, :].T.plot()


# %%
prices = y.iloc[1400, :].values
plt.plot(prices)

CAPACITY = 20
schedule = np.zeros(H)
schedule[prices.argmax()] = -CAPACITY
schedule[prices.argmin()] = CAPACITY
inventory = np.roll(np.cumsum(schedule), 1)

# %%
fig, ax1 = plt.subplots(figsize=(6, 5))

ax1.plot(
    prices,
    color="red",
    label="Price",
    # ds="steps-post",
)
ax2 = ax1.twinx()
ax2.bar(
    np.arange(H),
    schedule,
    color="gray",
    label="Schedule",
    alpha=0.5,
    width=1,
    align="edge",
)
ax2.plot(
    np.arange(H),
    inventory,
    color="blue",
    linestyle="--",
    label="State of Charge",
)
ax1.set_ylabel("Price [EUR/MWh]", color="red")
ax2.set_ylabel("Schedule / Inventory [MWh]", color="blue")
ax1.set_xlabel("Hour of the day")
ax1.set_ylim(20, 120)
ax2.set_ylim(-25, 25)
ax1.set_xlim(-0.5, H - 0.5)
ax2.axhline(0, color="black", linewidth=1, linestyle=":")

ax1.set_xticks(np.arange(H), np.arange(H))
ax1.set_xticklabels(np.arange(H), rotation=90)
ax1.set_yticks(np.arange(20, 121, 10))

ax1.grid(which="both", linestyle=":", linewidth=0.5)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

plt.title("Battery Optimization")
plt.tight_layout()
plt.savefig(
    "experiments/epf_germany/figures_presentation/battery_example.png",
    dpi=300,
    bbox_inches="tight",
)


# %%

M = 20
cov = y.cov().values

rand_noise = (
    st.norm(
        np.zeros(H),
        np.diag(cov) ** 0.5,
    )
    .rvs((M, H), random_state=1)
    .T
)
corr_noise = (
    st.multivariate_normal(
        mean=np.zeros(H),
        cov=cov,
    )
    .rvs(size=M, random_state=1)
    .T
)

rand_ens = rand_noise + prices[:, None]
corr_ens = corr_noise + prices[:, None]

plt.figure(figsize=(6, 5))
plt.plot(
    rand_ens,
    color="red",
    alpha=0.5,
)
plt.plot(
    prices,
    color="black",
    label="Mean Price",
    linewidth=2,
)

plt.ylabel("Price [EUR/MWh]")
plt.xlabel("Hour of the day")
plt.xticks(np.arange(H), np.arange(H), rotation=90)
plt.grid(which="both", linestyle=":", linewidth=0.5)
plt.tight_layout()
plt.savefig(
    "experiments/epf_germany/figures_presentation/ens_uncorrelated.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

plt.figure(figsize=(6, 5))
plt.plot(
    corr_ens,
    color="green",
    alpha=0.5,
    label="Correlated Noise",
)
plt.plot(
    prices,
    color="black",
    label="Price",
    linewidth=2,
)
plt.ylabel("Price [EUR/MWh]")
plt.xlabel("Hour of the day")
plt.xticks(np.arange(H), np.arange(H), rotation=90)
plt.grid(which="both", linestyle=":", linewidth=0.5)
plt.tight_layout()
plt.savefig(
    "experiments/epf_germany/figures_presentation/ens_correlated.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()


# %%

data = pd.read_csv(
    "experiments/epf_germany/de_prices.csv", index_col=0, parse_dates=True
)

vmin = -50
vmax = 150
n_bins_epex = 300
n_bins_relo = 100
n_bins_fund = 200
color = "royalblue"

# %%


fig, axs = plt.subplots(
    4,
    2,
    figsize=(8, 12),
    gridspec_kw={"width_ratios": [5, 1]},
    sharey="row",
    sharex="col",
)

# Price time series
data["Price"].plot(ax=axs[0, 0], linewidth=0.5, color=color, label="Price")
axs[0, 0].set_title("Electricity Price Time Series")
axs[0, 0].set_xlabel("Time")
axs[0, 0].set_ylabel("Price [EUR/MWh]")
axs[0, 0].grid(True, which="both", linestyle="--", linewidth=0.5)
axs[0, 0].set_ylim(vmin, vmax)

# Price histogram
axs[0, 1].hist(
    data["Price"].values,
    bins=n_bins_epex,
    orientation="horizontal",
    color=color,
    edgecolor=None,
)
axs[0, 1].set_title("Histogram")
axs[0, 1].set_xlabel("Count")
axs[0, 1].set_ylabel("Price")
axs[0, 1].set_ylim(vmin, vmax)
axs[0, 1].set_axisbelow(True)
axs[0, 1].grid(linestyle="--", linewidth=0.5)

# Residual load time series
data["Load_DA_Forecast"].div(1000).plot(
    ax=axs[1, 0], linewidth=0.5, color="darkorange", label="Load"
)
axs[1, 0].set_title("Residual Load Time Series")
axs[1, 0].set_xlabel("Time")
axs[1, 0].set_ylabel("Residual Load [GWh]")
axs[1, 0].grid(True, which="both", linestyle="--", linewidth=0.5)

# Residual load histogram
axs[1, 1].hist(
    data["Load_DA_Forecast"].div(1000),
    bins=n_bins_relo,
    orientation="horizontal",
    color="darkorange",
    edgecolor=None,
)
axs[1, 1].set_title("Histogram")
axs[1, 1].set_xlabel("Count")
axs[1, 1].set_axisbelow(True)
axs[1, 1].grid(linestyle="--", linewidth=0.5)

# Renewables forecast time series
data["Renewables_DA_Forecast"].div(1000).plot(
    ax=axs[2, 0], linewidth=0.5, color="deepskyblue", label="Renewable Production"
)
axs[2, 0].set_title("Renewables Forecast Time Series")
axs[2, 0].set_xlabel("Time")
axs[2, 0].set_ylabel("Forecast [GWh]")
axs[2, 0].grid(True, which="both", linestyle="--", linewidth=0.5)
axs[2, 0].legend()

# Renewables forecast histogram
axs[2, 1].hist(
    data["Renewables_DA_Forecast"].div(1000),
    bins=n_bins_relo,
    orientation="horizontal",
    color="deepskyblue",
    alpha=0.5,
    label="Renewables",
)
axs[2, 1].set_title("Histogram")
axs[2, 1].set_xlabel("Count")
axs[2, 1].set_axisbelow(True)
axs[2, 1].grid(linestyle="--", linewidth=0.5)

# Move fuel prices to the bottom row
data["TTF_Gas"].plot(ax=axs[3, 0], linewidth=0.5, color="seagreen", label="TTF Gas")
data["EUA"].plot(ax=axs[3, 0], linewidth=0.5, color="purple", label="EUA")
data["API2_Coal"].plot(ax=axs[3, 0], linewidth=0.5, color="brown", label="API2 Coal")
data["Brent_oil"].plot(ax=axs[3, 0], linewidth=0.5, color="gray", label="Oil")
axs[3, 0].set_title("Fuel Prices Time Series")
axs[3, 0].set_xlabel("Time")
axs[3, 0].set_ylabel("Price [EUR/MWh or EUR/tCO2]")
axs[3, 0].grid(True, which="both", linestyle="--", linewidth=0.5)
axs[3, 0].legend(ncols=2, loc="upper left")

axs[3, 1].hist(
    data["TTF_Gas"].values,
    bins=n_bins_fund,
    orientation="horizontal",
    color="seagreen",
    alpha=0.5,
    label="TTF Gas",
)
axs[3, 1].hist(
    data["EUA"].values,
    bins=n_bins_fund,
    orientation="horizontal",
    color="purple",
    alpha=0.5,
    label="EUA",
)
axs[3, 1].hist(
    data["API2_Coal"].values,
    bins=n_bins_fund,
    orientation="horizontal",
    color="brown",
    alpha=0.5,
    label="API2 Coal",
)
axs[3, 1].hist(
    data["Brent_oil"].values,
    bins=n_bins_fund,
    orientation="horizontal",
    color="gray",
    alpha=0.5,
    label="Brent Oil",
)
axs[3, 1].set_title("Histogram")
axs[3, 1].set_xlabel("Count")
axs[3, 1].set_axisbelow(True)
axs[3, 1].grid(linestyle="--", linewidth=0.5)

plt.tight_layout()
plt.savefig(
    "experiments/epf_germany/figures_presentation/data_overview.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# %%
