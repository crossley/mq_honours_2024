import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

d  = pd.read_csv('../data/sub_2_data.csv')

sns.scatterplot(x='x', y='y', hue='sub_task', style='cat',  data=d)
plt.show()
