import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dir_data = "../data"

d_rec = []

for file in os.listdir(dir_data):

    if file.endswith(".csv"):
        d = pd.read_csv(os.path.join(dir_data, file))
        d["block"] = np.floor(d["trial"] / 25).astype(int)
        d["acc"] = d["cat"] == d["resp"]
        d["cat"] = d["cat"].astype("category")
        d["phase"] = ["Learn"] * 300 + ["Intervention"] * 300 + ["Test"] * 299
        d_rec.append(d)

d = pd.concat(d_rec, ignore_index=True)

# fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(10, 5))
# sns.scatterplot(data=d, x="x", y="y", style="cat", ax=ax[0, 0])
# plt.tight_layout()
# plt.show()

# calculate accuracy for each block
dd = d.groupby(["experiment", "condition", "subject", "block",
                "phase"])["acc"].mean().reset_index()

fig, ax = plt.subplots(3, 2, squeeze=False, figsize=(12, 5))
sns.lineplot(data=dd[(dd["experiment"] == 1) & (dd["condition"] == "relearn")],
             x="block",
             y="acc",
             hue="phase",
             legend=False,
             ax=ax[0, 0])
sns.lineplot(data=dd[(dd["experiment"] == 1)
                     & (dd["condition"] == "new_learn")],
             x="block",
             y="acc",
             hue="phase",
             legend=False,
             ax=ax[0, 1])
sns.lineplot(data=dd[(dd["experiment"] == 2) & (dd["condition"] == "relearn")],
             x="block",
             y="acc",
             hue="phase",
             legend=False,
             ax=ax[1, 0])
sns.lineplot(data=dd[(dd["experiment"] == 2)
                     & (dd["condition"] == "new_learn")],
             x="block",
             y="acc",
             hue="phase",
             legend=False,
             ax=ax[1, 1])
sns.lineplot(data=dd[(dd["experiment"] == 3) & (dd["condition"] == "relearn")],
             x="block",
             y="acc",
             hue="phase",
             legend=False,
             ax=ax[2, 0])
sns.lineplot(data=dd[(dd["experiment"] == 3)
                     & (dd["condition"] == "new_learn")],
             x="block",
             y="acc",
             hue="phase",
             legend=False,
             ax=ax[2, 1])
ax[0, 0].set_title("Exp 1: Relearn")
ax[0, 1].set_title("Exp 1: New Learn")
ax[1, 0].set_title("Exp 1: Relearn")
ax[1, 1].set_title("Exp 1: New Learn")
ax[2, 0].set_title("Exp 1: Relearn")
ax[2, 1].set_title("Exp 1: New Learn")
plt.tight_layout()
plt.show()


d.groupby(["experiment", "condition"])["subject"].unique()
