# %%
# Allow running as notebook and as script
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# %%
# Import necessary packages
from functools import partial
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
import statsmodels.tsa.stattools as sm_tools
from const_and_helper import (  # MODEL_NAMES_MAPPING,
    FOLDER_DATA,
    FOLDER_FIGURES,
    FOLDER_RESULTS,
    FOLDER_TABLES,
    MODEL_NAMES_MAPPING,
    PLT_SAVE_OPTIONS,
    PLT_TEX_OPTIONS,
)
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams.update(PLT_TEX_OPTIONS)


# CMAP_TABLES = sns.color_palette("coolwarm", as_cmap=True)
CMAP_TABLES = LinearSegmentedColormap.from_list(
    "table_cmap", ["teal", "gainsboro", "indianred"]
)
CMAP_TABLES.set_bad(color="white", alpha=0)


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


# %% Load the score file.
file = np.load(file=os.path.join(FOLDER_RESULTS, "scores.npz"))
MODEL_NAMES = file["model_names"]
MODEL_NAMES_NICE = [MODEL_NAMES_MAPPING.get(name) for name in MODEL_NAMES]
N_MODELS = len(MODEL_NAMES)
THRESHOLD = 10000


def format_func(x, threshold=10000):
    """Format the outout and replace NaN with "" and large values with $>10^4$"""
    if pd.isna(x):
        return ""
    elif x > threshold:
        return f"$>{threshold:,}$"
    else:
        return f"{x:.3f}"


# %%
subsets = {
    "a": np.arange(N_TEST),
    "b": np.arange(736),
    "c": np.arange(736, N_TEST),
}

for subset_name, MASK in subsets.items():
    score_overview = pd.DataFrame(index=MODEL_NAMES)
    score_overview["RMSE"] = np.sqrt(np.mean(file["error_mean"][MASK] ** 2, (0, 2)))
    score_overview["MAE"] = np.mean(np.abs(file["error_med"][MASK]), (0, 2))
    score_overview["CRPS"] = np.mean(file["error_crps"][MASK], (0, 2))
    score_overview["$\\text{VS}_{p=0.5}$"] = file["vs05"][MASK].mean(0) / 24**2
    score_overview["$\\text{VS}_{p=1}$"] = file["vs10"][MASK].mean(0) / 24**2
    score_overview["ES"] = file["es"][MASK].mean(0)
    score_overview["DSS"] = file["dss"][MASK].mean(0)
    score_overview["LS"] = file["ls"][MASK].mean(0)
    score_overview["$\\text{MC}_{0.95}$"] = (
        np.mean(file["prediction_band_cover"][MASK], 0) - 0.95
    )
    score_overview["$\\text{JPBW}_{0.95}$"] = np.mean(
        file["prediction_band_width"][MASK], (0, 2)
    )
    score_overview = score_overview.rename(index=MODEL_NAMES_MAPPING)

    # Do the styling
    styled = score_overview.style.format(precision=3)

    for col in score_overview.columns:
        if col != "$\\text{MC}_{0.95}$":
            gmap = np.log(score_overview[col] + 1e-5)
        else:
            gmap = np.abs(score_overview[col] + 1e-5)

        gmap = gmap.clip(lower=0, upper=THRESHOLD)

        styled = styled.background_gradient(
            subset=[col], cmap=CMAP_TABLES, gmap=np.log(gmap + 1)
        )
        styled = styled.apply(
            lambda x, gmap=gmap: [
                (
                    "font-weight: bold; text-decoration: underline"
                    if v == gmap.idxmin()
                    else ""
                )
                for v in x.index
            ],
            subset=[col],
        )
        styled = styled.format(partial(format_func, threshold=THRESHOLD))
    styled = styled.map(lambda x: ("color: white;" if pd.isna(x) else ""))

    # Save the styled DataFrame to HTML
    # For nice display in presentations
    styled.to_latex(
        os.path.join(FOLDER_TABLES, f"scoringrules_{subset_name}.tex"),
        convert_css=True,
        hrules=True,
        column_format="R{6cm}" + "L{1.9cm}" * score_overview.shape[1],
    )
    styled.to_html(
        os.path.join(FOLDER_TABLES, f"scoringrules_{subset_name}.html"),
        escape=False,
    )


# %%
ls_dataframe = pd.DataFrame(
    file["ls"],
    index=df_X.index[IDX_TEST],
    columns=MODEL_NAMES_NICE,
)

skill_scores = 1 - ls_dataframe.div(ls_dataframe["oDistReg+GC"], axis=0)

yp_min = -0.5
yp_max = 0.1
ys_min = -100
ys_max = 800
y_steps = 7
line_styles = [":", "--", "-."] * 3

ax = (
    skill_scores.filter(
        regex="MvDistReg",
    )
    .rolling(182, min_periods=7)
    .mean()
    .plot(cmap="turbo", figsize=(8, 4), lw=2, legend=False)
)
for line, ls in zip(ax.get_lines(), line_styles):
    line.set_linestyle(ls)
plt.ylim(yp_min, yp_max)
plt.yticks(np.linspace(yp_min, yp_max, y_steps))
plt.ylabel("Skill Score (Log-Score relative to oDis+GC)")
prices.loc[skill_scores.index].mean(axis=1).plot(
    secondary_y=True,
    ax=plt.gca(),
    color="black",
    alpha=0.33,
    lw=1,
    label="Daily Base Price",
    zorder=0,
)
plt.ylim(ys_min, ys_max)
plt.yticks(np.linspace(ys_min, ys_max, y_steps))
plt.grid(which="both", ls=":")
plt.ylabel("Average Price (EUR/MWh)")
plt.xlabel("Date")

lines_primary, labels_primary = ax.get_legend_handles_labels()
lines_secondary, labels_secondary = ax.right_ax.get_legend_handles_labels()
all_lines = lines_primary + lines_secondary
all_labels = labels_primary + labels_secondary
unique = dict(zip(all_labels, all_lines))
plt.title("Skill Scores of Multivariate Models over Time")
plt.tight_layout()
plt.legend(
    unique.values(),
    unique.keys(),
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.5, -0.4),
)


plt.savefig(
    os.path.join(FOLDER_FIGURES, "skill_scores_timeseries.pdf"), **PLT_SAVE_OPTIONS
)
plt.savefig(
    os.path.join(FOLDER_FIGURES, "skill_scores_timeseries.png"), **PLT_SAVE_OPTIONS
)
plt.show(block=False)

# %%

plt.figure(figsize=(8, 4))
plt.bar(
    skill_scores.mean().dropna().index,
    skill_scores.mean().dropna(),
    color="teal",
    edgecolor="black",
)
plt.gca().set_axisbelow(True)
plt.grid(axis="y", ls=":")
plt.ylim(-0.2, 0.05)
plt.xticks(rotation=90)
plt.show(block=False)

# %%


# %%
# Create plots for RMSE and MAE by Hour
LS_MAPPING = {"LARX": ":", "oDis": "--", "oMvD": "-"}
LS = [LS_MAPPING[i[:4]] for i in MODEL_NAMES_NICE]

error_mean = file["error_mean"]
error_med = file["error_med"]
error_crps = file["error_crps"]

hour_rmse = np.mean(error_mean**2, 0) ** 0.5
hour_mae = np.mean(np.abs(error_med), 0)
hour_crps = np.mean(error_crps, 0)

fig, axes = plt.subplots(1, 3, figsize=(8, 4), sharex=True, dpi=300)
for i in range(3):
    axes[i].set_prop_cycle(
        color=plt.cm.turbo(np.linspace(0, 1, N_MODELS)),
        linestyle=LS,
    )


# RMSE by Hour
axes[0].set_title("RMSE by Hour")
axes[0].plot(hour_rmse.T, label=MODEL_NAMES_NICE)
axes[0].set_ylim(10, 65)
axes[0].set_ylabel(r"$\text{RMSE}_h$")

# MAE by Hour
axes[1].set_title("MAE by Hour")
axes[1].plot(hour_mae.T)
axes[1].set_ylim(5, 35)
axes[1].set_ylabel(r"$\text{MAE}_h$")

# CRPS by Hour
axes[2].set_title("CRPS by Hour")
axes[2].plot(hour_crps.T)
axes[2].set_ylim(5, 25)
axes[2].set_ylabel(r"$\text{CRPS}_h$")

for i in range(3):
    axes[i].set_xlabel(r"Delivery Hour $h$")
    axes[i].set_xticks(np.arange(H, step=2), np.arange(H, step=2), rotation=90)
    axes[i].grid(ls=":")
    axes[i].set_xlim(0, 23)

# Get legend handles and labels from axes[0] and create a joint legend
lines, labels = axes[0].get_legend_handles_labels()

# Remove duplicate labels
unique = dict(zip(labels, lines))
fig.legend(
    unique.values(),
    unique.keys(),
    loc="upper center",
    ncol=4,
    prop={"size": 8},
    bbox_to_anchor=(0.5, 1.2),
)
plt.tight_layout()
plt.savefig(os.path.join(FOLDER_FIGURES, "hourly_metrics.pdf"), **PLT_SAVE_OPTIONS)
plt.savefig(os.path.join(FOLDER_FIGURES, "hourly_metrics.png"), **PLT_SAVE_OPTIONS)
plt.show(block=False)

# %%
# Spike Analysis as requested by the reviewers
# We define a high intraday spreads as the 10% of the highest spreads
# in the day-ahead prices
# We define a spike as a price that is
# - above the 95% quantile of the maximum prices
# - below the 5% quantile of the minimum prices
# Define masks, and then calculate grouped average scores

error_crps = file["error_crps"].mean(-1)
error_ls = file["ls"]
prediction_band_cover = file["prediction_band_cover"]

spreads = prices_test.max(1) - prices_test.min(1)
mask_spread = np.where(spreads > np.quantile(spreads, 0.9))[0]
mask_spikes_h = np.where(prices_test.max(1) > np.quantile(prices_test.max(1), 0.95))[0]
mask_spikes_l = np.where(prices_test.min(1) < np.quantile(prices_test.min(1), 0.05))[0]
mask_spike = np.unique(np.stack((mask_spikes_h, mask_spikes_l)))

spike_analysis = pd.DataFrame(index=MODEL_NAMES)
spike_analysis[("CRPS", "Spread", "Low")] = error_crps[~mask_spread].mean(0)
spike_analysis[("CRPS", "Spread", "High")] = error_crps[mask_spread].mean(0)
spike_analysis[("CRPS", "Spike", "Low")] = error_crps[~mask_spike].mean(0)
spike_analysis[("CRPS", "Spike", "High")] = error_crps[mask_spike].mean(0)

spike_analysis[("LS", "Spread", "Low")] = error_ls[~mask_spread].mean(0)
spike_analysis[("LS", "Spread", "High")] = error_ls[mask_spread].mean(0)
spike_analysis[("LS", "Spike", "Low")] = error_ls[~mask_spike].mean(0)
spike_analysis[("LS", "Spike", "High")] = error_ls[mask_spike].mean(0)

spike_analysis[("$\\text{MC}_{0.95}$", "Spread", "Low")] = prediction_band_cover[
    ~mask_spread
].mean(0) - (1 - 0.05)
spike_analysis[("$\\text{MC}_{0.95}$", "Spread", "High")] = prediction_band_cover[
    mask_spread
].mean(0) - (1 - 0.05)
spike_analysis[("$\\text{MC}_{0.95}$", "Spike", "Low")] = prediction_band_cover[
    ~mask_spike
].mean(0) - (1 - 0.05)
spike_analysis[("$\\text{MC}_{0.95}$", "Spike", "High")] = prediction_band_cover[
    mask_spike
].mean(0) - (1 - 0.05)

spike_analysis.columns = pd.MultiIndex.from_tuples(spike_analysis.columns)
spike_analysis_gmap = spike_analysis.copy()
spike_analysis_gmap.round(2)
cols = ["$\\text{MC}_{0.95}$"]
spike_analysis_gmap.loc[:, cols] = np.abs(spike_analysis_gmap.loc[:, cols])
spike_analysis_rank = spike_analysis_gmap.rank().fillna(len(MODEL_NAMES) + 1)

spike_analysis_str = spike_analysis.map(lambda x: f"{x:.2f}" if not np.isnan(x) else "")
spike_analysis_str = (
    spike_analysis_str
    + " ("
    + spike_analysis_rank.astype(int).astype(str).map(lambda x: x.zfill(2))
    + ")"
)
spike_analysis_str = spike_analysis_str.rename(index=MODEL_NAMES_MAPPING)

# Maybe we can make the background gradient based on
# the change in the models ranking?
# TODO: Need to add the model names
style = spike_analysis_str.style
for col in spike_analysis.columns:
    style.background_gradient(
        gmap=np.log(spike_analysis_gmap[col].values),
        axis=0,
        subset=[col],
        cmap=CMAP_TABLES,
    )
# %%
style = style.map(lambda x: ("color: white;" if pd.isna(x) else ""))
style.to_latex(
    "experiments/epf_germany/tables/spike_analysis.tex",
    hrules=True,
    convert_css=True,
    multicol_align="c",
    column_format="l" + "r" * spike_analysis_rank.shape[1],
)

# %%
### DIEBOLD MARIANO TEST
N_MODELS = len(MODEL_NAMES)
dm_test = np.zeros((4, N_MODELS, N_MODELS))
adf_test = np.zeros((4, N_MODELS, N_MODELS))

# Names for the score plots
FULL_SCORE_NAMES = [
    "Variogram Score ($p=1$)",
    "Energy Score",
    "Dawid Sebastiani Score",
    "Log-Score",
]

scores = np.stack((file["vs10"], file["es"], file["dss"], file["ls"]), axis=-1)
for i, s in enumerate(FULL_SCORE_NAMES):
    for m1, m2 in product(range(N_MODELS), range(N_MODELS)):
        if m1 != m2:
            loss_differential = scores[:, m1, i] - scores[:, m2, i]

            dm_test[i, m1, m2] = st.ttest_1samp(
                loss_differential,
                0,
                alternative="greater",
            ).pvalue
            # Skip ADF Test for CP because CP has NANs for the
            # multivariate Loss functions
            if not np.any(np.isnan(loss_differential)):
                adf_test[i, m1, m2] = sm_tools.adfuller(
                    loss_differential,
                )[1]
        if m1 == m2:
            dm_test[i, m1, m2] = np.nan
            adf_test[i, m1, m2] = np.nan

# %%
# Define the colors
colors = [(0, "green"), (0.5, "yellow"), (0.75, "red"), (1, "black")]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

# This is the big DM matirx plot.
fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)
for i, ax in enumerate(axes.flatten()):
    sns.heatmap(
        dm_test[i].round(2),
        annot=True,
        cmap=cmap,
        vmax=0.1,
        square=True,
        alpha=1,
        ax=ax,
        linewidths=0.5,
        linecolor="grey",
        cbar=False,
        cbar_kws=(
            None
            if i < 3
            else {"label": "$p$-value for the Diebold-Mariano Test", "shrink": 0.5}
        ),
    )
    ax.set_title(FULL_SCORE_NAMES[i])
    ax.set_yticks(np.arange(0.5, N_MODELS))
    ax.set_yticklabels(MODEL_NAMES_NICE, rotation=0)
    ax.set_xticks(np.arange(0.5, len(MODEL_NAMES)))
    ax.set_xticklabels(MODEL_NAMES_NICE, rotation=90)

plt.tight_layout()
# Place the colorbar above all subplots, centered
cax = inset_axes(
    axes[0, 0],
    width="200%",
    height="10%",
    loc="upper left",
    bbox_to_anchor=(0.0, 0.75, 1, 0.5),
    bbox_transform=axes[0, 0].transAxes,
    borderpad=0,
)
norm = plt.Normalize(vmin=0, vmax=0.1)
cb = plt.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    cax=cax,
    orientation="horizontal",
    ticks=[0, 0.05, 0.1],
)
cb.set_label("$p$-value for the Diebold-Mariano Test")
plt.savefig(os.path.join(FOLDER_FIGURES, "dm_matrix.png"), **PLT_SAVE_OPTIONS)
plt.savefig(os.path.join(FOLDER_FIGURES, "dm_matrix.pdf"), **PLT_SAVE_OPTIONS)
plt.show(block=False)

# %%
# Score differentials are mostly stationary (for LS and DSS)
# For the VS and ES there are some pairs that are not stationary (only 5 pairs at alpha = 0.05 )
for i in range(scores.shape[2]):
    sns.heatmap((np.round(adf_test, 2))[i], annot=True, cmap="coolwarm", center=0)
    plt.show(block=False)

# %%
