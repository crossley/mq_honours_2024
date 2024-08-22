import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(10, 5))
sns.scatterplot(data=d[(d["condition"] == "4F4K_congruent")
                       & (d["sub_task"] == 1)],
                x="x",
                y="y",
                hue="cat",
                style="cat",
                ax=ax[0, 0])
sns.scatterplot(data=d[(d["condition"] == "4F4K_congruent")
                       & (d["sub_task"] == 2)],
                x="x",
                y="y",
                hue="cat",
                style="cat",
                ax=ax[0, 1])
sns.scatterplot(data=d[(d["condition"] == "4F4K_incongruent")
                       & (d["sub_task"] == 1)],
                x="x",
                y="y",
                hue="cat",
                style="cat",
                ax=ax[1, 0])
sns.scatterplot(data=d[(d["condition"] == "4F4K_incongruent")
                       & (d["sub_task"] == 2)],
                x="x",
                y="y",
                hue="cat",
                style="cat",
                ax=ax[1, 1])
ax[0, 0].set_title("4F4K_congruent_sub_task_1")
ax[0, 1].set_title("4F4K_congruent_sub_task_2")
ax[1, 0].set_title("4F4K_congruent_sub_task_1")
ax[1, 1].set_title("4F4K_congruent_sub_task_2")
plt.tight_layout()
plt.show()

# add a block column that split trials up into blocks of 25
d["block"] = np.floor(d["trial"] / 25).astype(int)

# calculate accuracy for each block
dd = d.groupby(["condition", "subject", "sub_task", "block"],
               observed=True)["acc"].mean().reset_index()

fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(8, 8))
sns.lineplot(data=dd,
             x="block",
             y="acc",
             style="sub_task",
             hue="condition",
             ax=ax[0, 0])
ax[0, 0].set_ylim(0, 1)
plt.tight_layout()
plt.show()

dd = dd.sort_values(
    by=["condition", "subject", "sub_task", "block"]).reset_index(drop=True)

dd.to_csv("../data_summary/summary.csv", index=False)
