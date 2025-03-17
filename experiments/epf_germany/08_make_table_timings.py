import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PATH_COPULA = "results/pred_copula.npz"
PATH_ARX = "results/pred_univariate_benchmark.npz"
PATH_DISTREG_UNI = "results/univariate_predictions_distreg.npz"
PATH_DISTREG_MUL = "results/pred_multivariate.npz"
KEY = "timings"

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

out = pd.DataFrame(index=MODELS)
out.index.name = "Model"
out["Initial Fit"] = timings[0, :]
out["Avg. Update"] = timings[1:, :].mean(0)
out["Std. Update"] = timings[1:, :].std(0)
out["Total Time"] = timings[:, :].sum(0)
out["Est. Speedup"] = "$\\times$" + np.floor(
    (out["Initial Fit"] * 736) / out["Total Time"]
).astype(int).astype(str)


styled = (
    out.reset_index()
    .style.format(
        precision=2, subset=["Initial Fit", "Avg. Update", "Std. Update", "Total Time"]
    )
    .format(precision=1, subset=["Est. Speedup"])
    .hide(axis="index")
)
styled.to_latex(
    "tables/timings.tex", convert_css=True, hrules=True, column_format="lrrrrr"
)
with open("tables/timings.html", "w") as f:
    f.write(styled.to_html(escape=True))


plot_data = pd.DataFrame(index=MODELS)
plot_data.index.name = "Model"
plot_data["Initial Fit"] = timings[0, :]
plot_data["All Updates"] = timings[1:, :].sum(0)


plt.figure(figsize=(6, 5))
plot_data.plot(
    kind="bar",
    stacked=True,
    edgecolor="black",
    color=["#8dd3c7", "#fb8072"],
    ax=plt.gca(),
)
plt.gca().set_axisbelow(True)
plt.yticks(np.arange(0, max(out["Total Time"]), 300))
plt.grid(axis="y")
plt.ylabel("Total Study Time (seconds)")
plt.xticks(rotation=25, ha="right", rotation_mode="anchor")
plt.xlabel(None)
plt.tight_layout()
plt.savefig("figures/computation_times.png", dpi=300)
plt.show()
