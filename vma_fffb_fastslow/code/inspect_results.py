from imports import *
from util_func import *

d_trl = pd.read_csv("../data/sub_1_data.csv")
d_mv = pd.read_csv("../data/sub_1_data_move.csv")

d_trl = d_trl.sort_values(["trial"])
d_mv = d_mv.sort_values(["t", "trial"])

d = pd.merge(d_mv, d_trl, how="outer", on=["condition", "subject", "trial"])

# TODO: Add phase
# rotation[n_trial // 3:2 * n_trial // 3] = 15 * np.pi / 180
n_trial = d["trial"].nunique()
d["phase"] = "adapt"
d.loc[d["trial"] < n_trial // 3, "phase"] = "base"
d.loc[d["trial"] > 2 * n_trial // 3, "phase"] = "wash"

d = d[d["state"].isin(["state_moving", "state_feedback_mp", "state_moving_2"])]

d["x"] = d["x"] - d["x"].values[0]
d["y"] = d["y"] - d["y"].values[0]
d["y"] = -d["y"]

d = d.groupby(["condition", "subject", "trial"],
              group_keys=False).apply(compute_kinematics)

fig, ax = plt.subplots(1, 3, squeeze=False)
sns.scatterplot(data=d, x="x", y="y", hue="su", ax=ax[0, 0])
sns.scatterplot(data=d, x="trial", y="imv", hue="su", ax=ax[0, 1])
sns.scatterplot(data=d, x="trial", y="emv", hue="su", ax=ax[0, 2])
plt.show()

dd = d.groupby(["condition", "subject", "phase", "trial", "su"],
               group_keys=False).apply(interpolate_movements)

fig, ax = plt.subplots(1, 3, squeeze=False)
sns.scatterplot(data=dd, x="x", y="y", hue="su", ax=ax[0, 0])
sns.scatterplot(data=dd, x="trial", y="imv", hue="su", ax=ax[0, 1])
sns.scatterplot(data=dd, x="trial", y="emv", hue="su", ax=ax[0, 2])
plt.show()

ddd = (dd.groupby(["condition", "subject", "phase", "su",
                   "relsamp"])[["t", "x", "y", "v"]].mean().reset_index())

fig, ax = plt.subplots(1, 1, squeeze=False)
sns.scatterplot(data=ddd, x="x", y="y", hue="su", style="phase", ax=ax[0, 0])
plt.show()

d.plot(subplots=True)
plt.show()
dd.plot(subplots=True)
plt.show()
ddd.plot(subplots=True)
plt.show()

# fig, ax = plt.subplots(2, 2, squeeze=False)
# sns.lineplot(
#     data=d[d["condition"] == "slow"],
#     x="trial",
#     y="imv",
#     hue="su",
#     #        style="phase",
#     markers=True,
#     legend=False,
#     ax=ax[0, 0],
# )
# sns.lineplot(
#     data=d[d["condition"] == "slow"],
#     x="trial",
#     y="emv",
#     hue="su",
#     #        style="phase",
#     markers=True,
#     legend=False,
#     ax=ax[0, 1],
# )
# sns.lineplot(
#     data=d[d["condition"] == "fast"],
#     x="trial",
#     y="imv",
#     hue="su",
#     #        style="phase",
#     markers=True,
#     legend=False,
#     ax=ax[1, 0],
# )
# sns.lineplot(
#     data=d[d["condition"] == "fast"],
#     x="trial",
#     y="emv",
#     hue="su",
#     #        style="phase",
#     markers=True,
#     legend=False,
#     ax=ax[1, 1],
# )
# [x.set_xlabel("Trial") for x in ax.flatten()]
# [x.set_ylabel("Initial Movement Vector") for x in ax[:, 0].flatten()]
# [x.set_ylabel("Endpoint Movement Vector") for x in ax[:, 1].flatten()]
# plt.tight_layout()
# plt.show()
