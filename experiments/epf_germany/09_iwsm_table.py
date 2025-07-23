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

PATH_COPULA = "results/pred_copula.npz"
PATH_ARX = "results/pred_univariate_benchmark.npz"
PATH_DISTREG_UNI = "results/univariate_predictions_distreg.npz"
PATH_DISTREG_MUL = "results/pred_multivariate.npz"
KEY = "timings"


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
MODELS = [
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


avg_scores = scores.mean(axis=0)
avg_scores = pd.DataFrame(avg_scores, columns=SCORE_NAMES)
avg_scores["Model"] = MODELS
avg_scores["RMSE"] = avg_scores["RMSE"] ** 0.5
avg_scores = avg_scores.set_index("Model")


timings_univ_distreg = np.load(PATH_DISTREG_UNI)[KEY]
timings_copula = np.load(file=PATH_COPULA)[KEY]
timings_copula = timings_copula + timings_univ_distreg.sum(axis=1, keepdims=True)
timings_mv_distreg = np.load(PATH_DISTREG_MUL)[KEY].T
timings_arx = np.load(PATH_ARX)[KEY]


timings = np.hstack(
    (
        timings_arx,
        timings_univ_distreg.sum(axis=1, keepdims=True),
        timings_copula,
        timings_mv_distreg,
    )
)


out = pd.DataFrame(index=MODELS)
out.index.name = "Model"
out["Initial Fit"] = timings[0, :]
out["Avg. Update"] = timings[1:, :].mean(0)
out["Std. Update"] = timings[1:, :].std(0)
out["Total Time"] = timings[:, :].sum(0)
out["Est. Speedup"] = "$\\times$" + np.floor(
    (out["Initial Fit"] * 736) / out["Total Time"]
).astype(int).astype(str)


df = pd.concat([avg_scores, out], axis=1)
df = df.reset_index()
df["Total Time"] = df["Total Time"] / 60


print(df)


COLS = [
    "RMSE",
    "CRPS",
    # "$\\text{VS}_{p=0.5}$",
    "$\\text{VS}_{p=1}$",
    "ES",
    "DSS",
    "LS",
    "Total Time",
]

style = (df.loc[:, ["Model"] + COLS].style).background_gradient(
    axis=0, subset=COLS, cmap="turbo"
)
style.format(precision=2, subset=COLS[:-1])
style.format(precision=1, subset=COLS[-1])
style.apply(
    lambda x: [
        "font-weight: bold; text-decoration: underline" if v == x.min() else ""
        for v in x
    ],
    axis=0,
    subset=COLS,
).hide(axis="index")


style.to_latex(
    "tables/summary_iwsm.tex",
    convert_css=True,
    hrules=True,
    column_format="l" + "r" * len(COLS),
)

with open("tables/summary_iwsm.html", "w") as f:
    f.write(style.to_html(escape=True))
