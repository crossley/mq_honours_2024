from imports import *
from util_func import *

# TODO: sub 0 data file is whack
d = load_all_data()
d = d.sort_values(["condition", "subject", "trial"])

d.loc[d["target_angle"] > 180, "target_angle"] -= 360
d.loc[(d["target_angle"] == 180) & (d["endpoint_theta"] < 0),
      "endpoint_theta"] += 360
d["endpoint_theta"] = -d["endpoint_theta"] + d["target_angle"]

d = d.groupby(["condition", "subject", "trial"],
              group_keys=False).apply(compute_kinematics, include_groups=True)

# dd = d.groupby(
#     ["condition", "subject", "phase", "trial", "target_angle"], group_keys=False
# ).apply(interpolate_movements)

# ddd = (
#     dd.groupby(["condition", "subject", "phase", "target_angle", "relsamp"])[
#         ["time", "x", "y", "v"]
#     ]
#     .mean()
#     .reset_index()
# )

dp = d.copy()
pp = dp[[
    "condition", "subject", "cycle_phase", "target_angle", "phase", "imv"
]].drop_duplicates().reset_index(drop=True)

# pp["target_angle"] = pp["target_angle"].astype('category')

ppp = pp[pp["phase"] == "generalisation"]

fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(6, 4))
sns.lineplot(data=ppp[ppp["condition"] == "low"],
             x="cycle_phase",
             y="imv",
             hue="target_angle",
             ax=ax[0, 0])
[
    ax[0, 0].axhline(y=-ta, color="black", linestyle="--")
    for ta in ppp["target_angle"].unique()
]
ax[0, 0].set_title("Low")
plt.show()

fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(7, 4))
sns.lineplot(data=ppp[ppp["condition"] == "low"],
             x="cycle_phase",
             y="imv",
             hue="target_angle",
             ax=ax[0, 0])
sns.lineplot(data=ppp[ppp["condition"] == "high"],
             x="cycle_phase",
             y="imv",
             hue="target_angle",
             legend=False,
             ax=ax[0, 1])
[
    ax[0, 0].axhline(y=-ta, color="black", linestyle="--")
    for ta in ppp["target_angle"].unique()
]
[
    ax[0, 1].axhline(y=-ta, color="black", linestyle="--")
    for ta in ppp["target_angle"].unique()
]
ax[0, 0].set_title("Low")
ax[0, 1].set_title("High")
[ax_.set_ylim(-180, 0) for ax_ in ax.flatten()]
legend = ax[0, 0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.15), ncol=4)
plt.show()

# # plot histogram coloured by target angle
# fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(7, 4))
# sns.histplot(data=ppp[ppp["condition"] == "low"],
#              x="imv",
#              hue="target_angle",
#              multiple="stack",
#              ax=ax[0, 0])
# sns.histplot(data=ppp[ppp["condition"] == "high"],
#              x="imv",
#              hue="target_angle",
#              multiple="stack",
#              legend=False,
#              ax=ax[0, 1])
# ax[0, 0].set_title("Low")
# ax[0, 1].set_title("High")
# [ax_.set_xlim(-180, 0) for ax_ in ax.flatten()]
# plt.show()

dp = d.copy()
pp = dp[["condition", "subject", "trial", "target_angle", "phase",
         "imv"]].drop_duplicates().reset_index(drop=True)
pp["target_angle"] = pp["target_angle"].astype("category")

pp.groupby(["condition", "target_angle"]).nunique()

dpp = pp.groupby(["condition", "trial", "target_angle",
                  "phase"]).mean().reset_index()
dpp["target_angle"] = dpp["target_angle"].astype("category")
dpp["phase"] = dpp["phase"].astype("category")

fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(7, 4))
sns.lineplot(data=dpp[dpp["condition"] == "low"],
             x="trial",
             y="imv",
             hue="target_angle",
             ax=ax[0, 0])
sns.lineplot(data=dpp[dpp["condition"] == "high"],
             x="trial",
             y="imv",
             hue="target_angle",
             legend=False,
             ax=ax[0, 1])
ax[0, 0].set_title("Low")
ax[0, 1].set_title("High")
[ax_.set_ylim(-180, 0) for ax_ in ax.flatten()]
legend = ax[0, 0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.15), ncol=4)
plt.show()

# dp = d.copy()
# dp = dp[dp["target_angle"] == 90]
# dp["target_angle"] = dp["target_angle"].astype("category")
# dp.loc[np.abs(dp["emv"] - 90) > 10, "emv"] = np.nan
# dp = dp.reset_index()
#
# dp.loc[dp["phase"] == "baseline_no_fb", "cycle_phase"] += 0
# dp.loc[dp["phase"] == "baseline_continuous_fb", "cycle_phase"] += 2
# dp.loc[dp["phase"] == "baseline_endpoint_fb", "cycle_phase"] += 4
# dp.loc[dp["phase"] == "baseline_mixed_fb", "cycle_phase"] += 6
# dp.loc[dp["phase"] == "generalisation", "cycle_phase"] += 8
# dp.loc[dp["phase"] == "washout_no_fb", "cycle_phase"] += 23
# dp.loc[dp["phase"] == "washout_fb", "cycle_phase"] += 25
#
# dp["subject"] = dp["subject"].astype("category")

# fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(7, 4))
# sns.lineplot(
#     data=dp,
#     x="cycle_phase",
#     y="imv",
#     hue="condition",
#     style="phase",
#     markers=True,
#     legend="brief",
#     ax=ax[0, 0],
# )
# legend = ax[0, 0].legend(loc="upper center",
#                          bbox_to_anchor=(0.5, 1.15),
#                          ncol=4)
# plt.show()
