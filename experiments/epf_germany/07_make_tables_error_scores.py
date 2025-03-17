from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

scores = np.load(file="results/scores.npy")
# Normalize the Variogram Score
scores[:, :, 4] = np.power(scores[:, :, 4] / 24**2, 0.5)
scores[:, :, 5] = np.power(scores[:, :, 5] / 24**2, 0.5)


N_MODELS = scores.shape[1]
N_BENCHMARKS = 4  # N_COPULA + N_MODELS_AR
SCORE_NAMES = [
    "RMSE",
    "MAE",
    "L2 Loss",
    "CRPS",
    "$\\text{VS}_{p=1}$",
    "$\\text{VS}_{p=0.5}$",
    "ES",
    "DSS",
    "LS",
]

avg_scores = scores.mean(axis=0)
avg_scores = pd.DataFrame(avg_scores, columns=SCORE_NAMES)
avg_scores["Model"] = [
    "LARX + N(0, $\sigma$)",
    "LARX + N(0, $\Sigma$)",
    "oDistReg",
    "oDistReg+GC",
    "oDistReg+spGC",
    "oMvDistReg(t, CD, OLS, ind)",
    "oMvDistReg(t, LRA, OLS, ind)",
    "oMvDistReg(t, CD, OLS)",
    "oMvDistReg(t, LRA, OLS)",
    "oMvDistReg(t, CD, LASSO)",
    "oMvDistReg(t, LRA, LASSO)",
]
avg_scores["RMSE"] = avg_scores["RMSE"] ** 0.5

scores_chosen = [
    "RMSE",
    "MAE",
    "CRPS",
    "$\\text{VS}_{p=0.5}$",
    "$\\text{VS}_{p=1}$",
    "ES",
    "DSS",
    "LS",
]
style = (avg_scores.loc[:, ["Model"] + scores_chosen].style).background_gradient(
    axis=0, subset=scores_chosen, cmap="turbo"
)
style.format(precision=3)
style.apply(
    lambda x: [
        "font-weight: bold; text-decoration: underline" if v == x.min() else ""
        for v in x
    ],
    axis=0,
    subset=scores_chosen,
).hide(axis="index")

style.to_latex("tables/scoringrules.tex", convert_css=True, hrules=True)

with open("tables/scoringrules.html", "w") as f:
    f.write(style.to_html(escape=False))


### DIEBOLD MARIANO TEST

dm_test = np.zeros((4, N_MODELS, N_MODELS))

# Names for the score plots
FULL_SCORE_NAMES = [
    "Variogram Score ($p=1$)",
    "Energy Score",
    "Dawid Sebastiani Score",
    "Log-Score",
]
indices = [4, 6, 7, 8]

for i, s in enumerate(indices):
    for m1, m2 in product(range(N_MODELS), range(N_MODELS)):
        if m1 != m2:
            dm_test[i, m1, m2] = st.ttest_1samp(
                scores[:, m1, s] - scores[:, m2, s],
                0,
                alternative="greater",
            ).pvalue
        if m1 == m2:
            dm_test[i, m1, m2] = np.nan

# Define the colors
colors = [(0, "green"), (0.5, "yellow"), (0.75, "red"), (1, "black")]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

names = avg_scores["Model"].tolist()

fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=True, sharey=True)
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
        cbar_kws={"label": "$p$-value for the Diebold-Mariano Test", "shrink": 0.5},
    )
    ax.set_title(FULL_SCORE_NAMES[i])
    ax.set_yticks(np.arange(0.5, len(names)))
    ax.set_yticklabels(names, rotation=0, ha="right", rotation_mode="anchor")
    ax.set_xticks(np.arange(0.5, len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", rotation_mode="anchor")

plt.tight_layout()
plt.savefig("figures/dm_matrix.png", dpi=300)
plt.show()
