import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import pingouin as pg
from util_func_dbm import *

dir_data = "../data"

d_rec = []

for f in os.listdir(dir_data):
    if f.endswith(".csv"):
        d = pd.read_csv(os.path.join(dir_data, f))
        d_rec.append(d)

d = pd.concat(d_rec)

print(d.groupby(["condition"])["subject"].unique())
print(d.groupby(["condition"])["subject"].nunique())

d["acc"] = d["cat"] == d["resp"]

d["cat"] = d["cat"].astype("category")
d["sub_task"] = d["sub_task"].astype("category")

# recode cat level names
d["cat"] = d["cat"].cat.rename_categories({
    107: "L1",
    97: "R1",
    108: "L2",
    115: "R2"
})

block_size = 100
d["block"] = d.groupby(["condition", "subject"]).cumcount() // block_size

d = d.sort_values(["condition", "subject", "block", "trial"])

d["block"].plot(subplots=True)
plt.show()

models = [
    nll_unix,
    nll_unix,
    nll_uniy,
    nll_uniy,
    nll_glc,
    nll_glc,
    nll_gcc_eq,
    nll_gcc_eq,
    nll_gcc_eq,
    nll_gcc_eq,
]
side = [0, 1, 0, 1, 0, 1, 0, 1, 2, 3]
k = [2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
n = block_size
model_names = [
    "nll_unix_0",
    "nll_unix_1",
    "nll_uniy_0",
    "nll_uniy_1",
    "nll_glc_0",
    "nll_glc_1",
    "nll_gcc_eq_0",
    "nll_gcc_eq_1",
    "nll_gcc_eq_2",
    "nll_gcc_eq_3",
]


def assign_best_model(x):
    model = x["model"].to_numpy()
    bic = x["bic"].to_numpy()
    best_model = np.unique(model[bic == bic.min()])[0]
    x["best_model"] = best_model
    return x


if not os.path.exists("../dbm_fits/dbm_results.csv"):
    dbm = (d.groupby(["condition", "subject", "sub_task",
                      "block"]).apply(fit_dbm, models, side, k, n,
                                      model_names).reset_index())

    dbm.to_csv("../dbm_fits/dbm_results.csv")

    dbm = dbm.groupby(["condition", "subject", "sub_task",
                       "block"]).apply(assign_best_model)

else:
    dbm = pd.read_csv("../dbm_fits/dbm_results.csv")

dd = dbm.loc[dbm["model"] == dbm["best_model"]]
ddd = dd[["condition", "subject", "sub_task", "block",
          "best_model"]].drop_duplicates()
dcat = d[["sub_task", "x", "y", "cat"]].drop_duplicates()

fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(12, 8))
sns.scatterplot(data=dcat[dcat["sub_task"] == 0],
                x="x",
                y="y",
                hue="cat",
                ax=ax[0, 0])
sns.scatterplot(data=dcat[dcat["sub_task"] == 1],
                x="x",
                y="y",
                hue="cat",
                ax=ax[0, 1])
ax[0, 0].get_legend().remove()
ax[0, 1].get_legend().remove()

for s in dd["subject"].unique():
    x = dd.loc[(dd["subject"] == s) & (dd["sub_task"] == 0) &
               (dd["block"] == 5)]

    best_model = x["best_model"].to_numpy()[0]

    if best_model in ("nll_unix_0", "nll_unix_1"):
        xc = x["p"].to_numpy()[0]
        ax[0, 0].plot([xc, xc], [0, 100], "--k")

    elif best_model in ("nll_uniy_0", "nll_uniy_1"):
        yc = x["p"].to_numpy()[0]
        ax[0, 0].plot([0, 100], [yc, yc], "--k")

    elif best_model in ("nll_glc_0", "nll_glc_1"):
        # a1 * x + a2 * y + b = 0
        # y = -(a1 * x + b) / a2
        a1 = x["p"].to_numpy()[0]
        a2 = np.sqrt(1 - a1**2)
        b = x["p"].to_numpy()[1]
        ax[0, 0].plot([0, 100], [-b / a2, -(100 * a1 + b) / a2], "-k")

    elif best_model in ("nll_gcc_eq_0", "nll_gcc_eq_1", "nll_gcc_eq_2",
                        "nll_gcc_eq_3"):
        xc = x["p"].to_numpy()[0]
        yc = x["p"].to_numpy()[1]
        ax[0, 0].plot([xc, xc], [0, 100], "-k")
        ax[0, 0].plot([0, 100], [yc, yc], "-k")

    ax[0, 0].set_xlim(-5, 105)
    ax[0, 0].set_ylim(-5, 105)

    x = dd.loc[(dd["subject"] == s) & (dd["sub_task"] == 1) &
               (dd["block"] == 5)]

    best_model = x["best_model"].to_numpy()[0]

    if best_model in ("nll_unix_0", "nll_unix_1"):
        xc = x["p"].to_numpy()[0]
        ax[0, 1].plot([xc, xc], [0, 100], "--k")

    elif best_model in ("nll_uniy_0", "nll_uniy_1"):
        yc = x["p"].to_numpy()[0]
        ax[0, 1].plot([0, 100], [yc, yc], "--k")

    elif best_model in ("nll_glc_0", "nll_glc_1"):
        # a1 * x + a2 * y + b = 0
        # y = -(a1 * x + b) / a2
        a1 = x["p"].to_numpy()[0]
        a2 = np.sqrt(1 - a1**2)
        b = x["p"].to_numpy()[1]
        ax[0, 1].plot([0, 100], [-b / a2, -(100 * a1 + b) / a2], "-k")

    elif best_model in ("nll_gcc_eq_0", "nll_gcc_eq_1", "nll_gcc_eq_2",
                        "nll_gcc_eq_3"):
        xc = x["p"].to_numpy()[0]
        yc = x["p"].to_numpy()[1]
        ax[0, 1].plot([xc, xc], [0, 100], "-k")
        ax[0, 1].plot([0, 100], [yc, yc], "-k")

    ax[0, 1].set_xlim(-5, 105)
    ax[0, 1].set_ylim(-5, 105)

# sns.countplot(data=ddd[ddd['sub_task'] == 0],
#               x='block',
#               hue='best_model',
#               ax=ax[1, 0])
# sns.countplot(data=ddd[ddd['sub_task'] == 1],
#               x='block',
#               hue='best_model',
#               ax=ax[1, 1])
# ax[1, 0].legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
# ax[1, 1].legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
# plt.show()
plt.tight_layout()
plt.savefig("../figures/fig_dbm_" + str(c) + ".pdf")
