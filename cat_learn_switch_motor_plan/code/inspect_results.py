import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

d  = pd.read_csv('../data/sub_3_data.csv')

d["acc"] = d["cat"] == d["resp"]

fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(10, 5))
sns.scatterplot(data=d, x='x', y='y', hue='sub_task', style='cat', ax=ax[0, 0])
plt.show()

# add a block column that split trials up into blocks of 25
d['block'] = np.floor(d['trial'] / 25).astype(int)

# calculate accuracy for each block
dd = d.groupby(['subject', 'sub_task', 'block'])["acc"].mean().reset_index()

fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(10, 5))
sns.lineplot(data=dd, x='block', y='acc', hue='sub_task', ax=ax[0, 0])
plt.show()
