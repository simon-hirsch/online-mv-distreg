# %%
# Allow running as notebook and as script
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from const_and_helper import (
    FOLDER_FIGURES,
    FOLDER_RESULTS,
    FOLDER_TABLES,
    MODEL_NAMES_MAPPING,
    PLT_SAVE_OPTIONS,
    PLT_TEX_OPTIONS,
)

plt.rcParams.update(PLT_TEX_OPTIONS)

# %%

PATH_COPULA = os.path.join(FOLDER_RESULTS, "pred_copula.npz")
PATH_ARX = os.path.join(FOLDER_RESULTS, "pred_univariate_benchmark.npz")
PATH_DISTREG_UNI = os.path.join(FOLDER_RESULTS, "univariate_predictions_distreg.npz")
PATH_DISTREG_MUL = os.path.join(FOLDER_RESULTS, "pred_multivariate.npz")
PATH_CONFORMAL = os.path.join(FOLDER_RESULTS, "benchmark_cp.npz")
PATH_GARCH = os.path.join(FOLDER_RESULTS, "pred_garch_benchmark.npz")
KEY = "timings"

timings_univ_distreg = np.load(PATH_DISTREG_UNI)[KEY]
timings_copula = np.load(file=PATH_COPULA)[KEY]
timings_copula = timings_copula + timings_univ_distreg.sum(axis=1, keepdims=True)
timings_mv_distreg = np.load(PATH_DISTREG_MUL)[KEY].T
timings_arx = np.load(PATH_ARX)[KEY]
timings_cp = np.load(PATH_CONFORMAL)[KEY]
timings_garch = np.load(PATH_GARCH)[KEY]


timings = np.hstack(
    (
        timings_arx,
        timings_arx[:, [0]],
        timings_garch[:, None],
        timings_univ_distreg.sum(axis=1, keepdims=True),
        timings_copula,
        timings_mv_distreg,
    )
)

# %%
file = np.load(file=os.path.join(FOLDER_RESULTS, "scores.npz"))
MODEL_NAMES = file["model_names"]
MODEL_NAMES_NICE = [MODEL_NAMES_MAPPING.get(name) for name in MODEL_NAMES]
N_MODELS = len(MODEL_NAMES)


N_TEST = timings.shape[0]

# %%
gmap = pd.DataFrame(index=MODEL_NAMES_NICE)
gmap.index.name = "Model"
gmap["Initial Fit"] = timings[0, :]
gmap["Avg. Update"] = timings[1:, :].mean(0)
gmap["Total Time"] = timings[:, :].sum(0)
gmap["Est. Speedup"] = np.floor(
    (gmap["Initial Fit"] * N_TEST) / gmap["Total Time"]
).astype(int)

out = gmap.copy(deep=True)
out["Est. Speedup"] = "$\\times$ " + out["Est. Speedup"].astype(str)

# %%
COLS = ["Initial Fit", "Avg. Update", "Total Time"]
styled = out.reset_index().style.format(
    precision=2, subset=["Initial Fit", "Avg. Update", "Total Time"]
)
styled = styled.format(precision=1, subset=["Est. Speedup"])
styled = styled.hide(axis="index")
for col in COLS + ["Est. Speedup"]:
    styled.background_gradient(
        cmap="Blues", subset=pd.IndexSlice[:, [col]], gmap=np.log(gmap[col].values)
    )
styled
# %%
styled.to_latex(
    os.path.join(FOLDER_TABLES, "timings.tex"),
    convert_css=True,
    hrules=True,
    column_format="lrrrrr",
)
with open(os.path.join(FOLDER_TABLES, "timings.html"), "w") as f:
    f.write(styled.to_html(escape=True))


# %%
# Plot for the timings
plot_data = pd.DataFrame(index=MODEL_NAMES_NICE)
plot_data.index.name = "Model"
plot_data["Initial Fit"] = timings[0, :]
plot_data["All Updates"] = timings[1:, :].sum(0)
plot_data = plot_data.div(plot_data.sum(axis=1), axis=0)

plt.figure(figsize=(4, 5))
plot_data.plot(
    kind="bar",
    stacked=True,
    edgecolor="black",
    color=["#8dd3c7", "#fb8072"],
    ax=plt.gca(),
)
plt.gca().set_axisbelow(True)
plt.ylim(0, 1)
plt.grid(axis="y")
plt.ylabel("Share")
plt.xticks(rotation=70, ha="right", rotation_mode="anchor")
plt.xlabel(None)
plt.tight_layout()
plt.savefig(os.path.join(FOLDER_FIGURES, "computation_times.pdf"), **PLT_SAVE_OPTIONS)
plt.savefig(os.path.join(FOLDER_FIGURES, "computation_times.png"), **PLT_SAVE_OPTIONS)
plt.show(block=False)

# %%
