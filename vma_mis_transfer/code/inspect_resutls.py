from imports import *
from util_func import *

d, dd, ddd = load_data()

d.groupby(["phase"])["rotation"].unique()

d.groupby(["condition"])["subject"].unique()

dp = d.groupby(
    ["condition", "session", "phase", "trial", "target_angle", "rotation"],
    observed=True)[["imv", "emv"]].mean().reset_index()

dp.sort_values(["condition", "session", "trial"], inplace=True)

dp.loc[dp["imv"] > 160, "imv"] -= 360
dp.loc[dp["emv"] > 160, "emv"] -= 360

dp["imv"] = dp["imv"] - dp["target_angle"]
dp["emv"] = dp["emv"] - dp["target_angle"]

dp["target_angle"] = dp["target_angle"].astype("category")

dpp = ddd[ddd["phase"] == "generalization"].groupby(
    ["condition", "session", "phase", "target_angle",
     "relsamp"])[["x", "y"]].mean().reset_index()
dpp["training_target"] = False
dpp.loc[dpp["target_angle"] == 0, "training_target"] = True
dpp["target_angle"] = dpp["target_angle"].astype("category")
dpp["training_target"] = dpp["training_target"].astype("category")

# NOTE: main results figure
fig, ax = plt.subplots(3, 3, squeeze=False)
fig.set_size_inches(11, 8)
fig.subplots_adjust(hspace=0.4, wspace=0.25)
fig.subplots_adjust(left=0.1, right=0.85)
fig.subplots_adjust(top=0.1, bottom=0.1)

for i, s in enumerate(dp.session.unique()):

    # initial movement vector across trials
    sns.scatterplot(
        data=dp[dp["session"] == s],
        x="trial",
        y="emv",
        hue="target_angle",
        markers=True,
        legend=False,
        ax=ax[i, 0],
    )

    ax[i, 0].plot(dp[dp["session"] == 1].trial,
                  dp[dp["session"] == 1].rotation * 2,
                  color="black",
                  linestyle="-")

    ax[i, 0].set_ylim(-100, 100)
    ax[i, 0].set_title("Session " + str(s))
    ax[i, 0].set_ylabel("Endpoint Movement Vector")

    # generalisation function
    sns.barplot(data=dp[(dp["phase"] == "generalization")
                        & (dp["session"] == s)],
                x="target_angle",
                y="emv",
                hue="target_angle",
                legend=False,
                ax=ax[i, 1])
    ax[i, 1].set_ylim(-150, 150)
    ax[i, 1].set_xlabel("Target Angle")
    ax[i, 1].set_ylabel("Endpoint Movement Vector")
    for label in ax[i, 1].get_xticklabels():
        label.set_rotation(45)

    # generalisation trajectories
    sns.scatterplot(
        data=dpp[dpp["session"] == s],
        x="x",
        y="y",
        hue="target_angle",
        style="training_target",
        markers=True,
        ax=ax[i, 2],
    )
    for ta in dpp.target_angle.unique():
        ta = ta + 90
        ax[i, 2].plot([0, 200 * np.cos(ta * np.pi / 180)],
                      [0, 200 * np.sin(ta * np.pi / 180)],
                      color="black",
                      linestyle="--")

ax[0, 2].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax[1, 2].legend().remove()
ax[2, 2].legend().remove()

plt.show()
