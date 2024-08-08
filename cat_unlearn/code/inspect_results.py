import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

d = pd.read_csv("../data/sub_2_data.csv")

d.condition.unique()

d["acc"] = d["cat"] == d["resp"]
d["cat"] = d["cat"].astype("category")

fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(10, 5))
sns.scatterplot(data=d, x="x", y="y", style="cat", ax=ax[0, 0])
plt.tight_layout()
plt.show()

# add a block column that split trials up into blocks of 25
d["block"] = np.floor(d["trial"] / 50).astype(int)

# calculate accuracy for each block
dd = d.groupby(["experiment", "condition", "subject",
                "block"])["acc"].mean().reset_index()

fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(10, 5))
sns.lineplot(data=dd[dd["experiment"] == 1], x="block", y="acc", ax=ax[0, 0])
plt.tight_layout()
plt.show()

d.groupby(["experiment", "condition", "block",
           "subject"])["resp"].nunique().reset_index()
