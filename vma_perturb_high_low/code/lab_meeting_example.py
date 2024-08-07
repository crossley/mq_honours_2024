import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

d = pd.read_csv("../data/data_trials_10.csv")

print(d.columns)

d["target_angle"] = d["target_angle"].astype("category") 
sns.scatterplot(x="trial", y="endpoint_theta", hue="target_angle", data=d)
plt.show()
