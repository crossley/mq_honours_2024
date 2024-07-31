import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

d1 = pd.read_csv('../data/sub_1_data.csv')
d2 = pd.read_csv('../data/sub_2_data.csv')
d3 = pd.read_csv('../data/sub_3_data.csv')
d4 = pd.read_csv('../data/sub_4_data.csv')

d1.condition.unique()
d2.condition.unique()
d3.condition.unique()
d4.condition.unique()

d = pd.concat([d1, d2, d3, d4], ignore_index=True)

d["acc"] = d["cat"] == d["resp"]

d["cat"] = d["cat"].astype("category")
d["sub_task"] = d["sub_task"].astype("category")

fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(10, 5))
sns.scatterplot(data=d[d["condition"] == '2F2K_congruent'],
                x='x',
                y='y',
                hue='sub_task',
                style='cat',
                ax=ax[0, 0])
sns.scatterplot(data=d[d["condition"] == '2F2K_incongruent'],
                x='x',
                y='y',
                hue='sub_task',
                style='cat',
                ax=ax[0, 1])
sns.scatterplot(data=d[d["condition"] == '4F4K_congruent'],
                x='x',
                y='y',
                hue='sub_task',
                style='cat',
                ax=ax[1, 0])
sns.scatterplot(data=d[d["condition"] == '4F4K_incongruent'],
                x='x',
                y='y',
                hue='sub_task',
                style='cat',
                ax=ax[1, 1])
ax[0, 0].set_title('2F2K_congruent')
ax[0, 1].set_title('2F2K_incongruent')
ax[1, 0].set_title('4F4K_congruent')
ax[1, 1].set_title('4F4K_incongruent')
plt.tight_layout()
plt.show()

# add a block column that split trials up into blocks of 25
d['block'] = np.floor(d['trial'] / 25).astype(int)

# calculate accuracy for each block
dd = d.groupby(['condition', 'subject', 'sub_task',
                'block'])["acc"].mean().reset_index()

fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(10, 5))
sns.lineplot(data=dd[dd["condition"] == "2F2K_congruent"],
             x='block',
             y='acc',
             hue='sub_task',
             ax=ax[0, 0])
sns.lineplot(data=dd[dd["condition"] == "2F2K_incongruent"],
             x='block',
             y='acc',
             hue='sub_task',
             ax=ax[0, 1])
sns.lineplot(data=dd[dd["condition"] == "4F4K_congruent"],
             x='block',
             y='acc',
             hue='sub_task',
             ax=ax[1, 0])
sns.lineplot(data=dd[dd["condition"] == "4F4K_incongruent"],
             x='block',
             y='acc',
             hue='sub_task',
             ax=ax[1, 1])
ax[0, 0].set_title('2F2K_congruent')
ax[0, 1].set_title('2F2K_incongruent')
ax[1, 0].set_title('4F4K_congruent')
ax[1, 1].set_title('4F4K_incongruent')
plt.tight_layout()
plt.show()
