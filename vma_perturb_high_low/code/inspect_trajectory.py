from imports import *
from util_func import *

# TODO: sub 0 data file is whack
d = load_all_data()
d = d.sort_values(["condition", "subject", "trial"])

# d.loc[d["target_angle"] > 180, "target_angle"] -= 360
# d.loc[(d["target_angle"] == 180) & (d["endpoint_theta"] < 0), "endpoint_theta"] += 360
# d["endpoint_theta"] = -d["endpoint_theta"] + d["target_angle"]

d = d.groupby(["condition", "subject", "trial"], group_keys=False).apply(
    compute_kinematics
)

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
dp = dp[dp["target_angle"] == 90]
dp["target_angle"] = dp["target_angle"].astype("category")
dp.loc[np.abs(dp["emv"] - 90) > 10, "emv"] = np.nan
dp = dp.reset_index()

dp.loc[dp["phase"] == "baseline_no_fb", "cycle_phase"] += 0
dp.loc[dp["phase"] == "baseline_continuous_fb", "cycle_phase"] += 2
dp.loc[dp["phase"] == "baseline_endpoint_fb", "cycle_phase"] += 4
dp.loc[dp["phase"] == "baseline_mixed_fb", "cycle_phase"] += 6
dp.loc[dp["phase"] == "generalisation", "cycle_phase"] += 8
dp.loc[dp["phase"] == "washout_no_fb", "cycle_phase"] += 23
dp.loc[dp["phase"] == "washout_fb", "cycle_phase"] += 25

dp["subject"] = dp["subject"].astype("category")

fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(7, 4))
sns.lineplot(
    data=dp,
    x="cycle_phase",
    y="emv",
    hue="phase",
    style="subject",
    markers=True,
    legend="brief",
    ax=ax[0, 0],
)
legend = ax[0, 0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=4)
plt.show()
