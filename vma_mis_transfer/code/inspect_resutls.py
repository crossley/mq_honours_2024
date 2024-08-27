import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import CubicSpline


def interpolate_movements(d):
    t = d["t"].to_numpy()
    x = d["x"].to_numpy()
    y = d["y"].to_numpy()
    v = d["v"].to_numpy()

    xs = CubicSpline(t, x)
    ys = CubicSpline(t, y)
    vs = CubicSpline(t, v)

    tt = np.linspace(t.min(), t.max(), 100)
    xx = xs(tt)
    yy = ys(tt)
    vv = vs(tt)

    relsamp = np.arange(0, tt.shape[0], 1)

    dd = pd.DataFrame({"relsamp": relsamp, "t": tt, "x": xx, "y": yy, "v": vv})
    dd["condition"] = d["condition"].unique()[0]
    dd["subject"] = d["subject"].unique()[0]
    dd["session"] = d["session"].unique()[0]
    dd["trial"] = d["trial"].unique()[0]
    dd["target_angle"] = d["target_angle"].unique()[0]
    dd["phase"] = d["phase"].unique()[0]
    dd["su"] = d["su"].unique()[0]
    dd["imv"] = d["imv"].unique()[0]
    dd["emv"] = d["emv"].unique()[0]

    return dd


def compute_kinematics(d):
    t = d["t"].to_numpy()
    x = d["x"].to_numpy()
    y = d["y"].to_numpy()

    x = x - x[0]
    y = y - y[0]
    y = -y

    r = np.sqrt(x**2 + y**2)
    theta = (np.arctan2(y, x)) * 180 / np.pi

    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    v = np.sqrt(vx**2 + vy**2)

    v_peak = v.max()
    # ts = t[v > (0.05 * v_peak)][0]
    ts = t[r > 0.1 * r.max()][0]

    imv = theta[(t >= ts) & (t <= ts + 0.1)].mean()
    emv = theta[-1]

    d["x"] = x
    d["y"] = y
    d["v"] = v
    d["imv"] = 90 - imv
    d["emv"] = 90 - emv

    return d


dir_data = "../data/"

d_rec = []
dd_rec = []
ddd_rec = []

# NOTE: useful when visually inspecting the data at the
# subject level
plot_sub_trajectories = False

sub_nums = [1, 3, 4, 5, 6, 7, 8, 9, 10]

for s in sub_nums:

    f_trl_1 = "sub_{}_data.csv".format(s)
    f_mv_1 = "sub_{}_data_move.csv".format(s)

    f_trl_2 = "sub_{}{}_data.csv".format(s, s)
    f_mv_2 = "sub_{}{}_data_move.csv".format(s, s)

    d_trl_1 = pd.read_csv(os.path.join(dir_data, f_trl_1))
    d_mv_1 = pd.read_csv(os.path.join(dir_data, f_mv_1))

    d_trl_2 = pd.read_csv(os.path.join(dir_data, f_trl_2))
    d_mv_2 = pd.read_csv(os.path.join(dir_data, f_mv_2))

    d_trl_1["session"] = 1
    d_mv_1["session"] = 1

    d_trl_2["session"] = 2
    d_mv_2["session"] = 2

    d_trl_2["subject"] = d_trl_1["subject"].unique()[0]
    d_mv_2["subject"] = d_mv_1["subject"].unique()[0]

    d_trl = pd.concat([d_trl_1, d_trl_2])
    d_mv = pd.concat([d_mv_1, d_mv_2])

    d_trl = d_trl.sort_values(["condition", "subject", "session", "trial"])
    d_mv = d_mv.sort_values(["condition", "subject", "session", "trial", "t"])
    d_hold = d_mv[d_mv["state"].isin(["state_holding"])]
    x_start = d_hold.x.mean()
    y_start = d_hold.y.mean()

    d_mv = d_mv[d_mv["state"].isin(["state_moving"])]

    # TODO: this needs to reflect the actual number of sessions
    phase = np.empty(d_trl.shape[0] // 2, dtype="object")

    # The experiment began with a familiarization phase of 33
    # reach trials (3 trials per target in pseudorandom order)
    # with continuous veridical visual feedback provided
    # throughout the reach.
    phase[:33] = "familiarisation"

    # The baseline phase consisted of 198 reach trials across
    # all 11 target directions (18 trials per target). On each
    # trial, the location of the target was randomized across
    # participants. For 2/3 of the reaches (132 trials),
    # continuous veridical cursor feedback was provided
    # throughout the trial. For the remaining 1/3 (66 trials),
    # visual feedback was completely withheld (i.e., no feedback
    # was given during the reach and no feedback was given at
    # the end of the reach about reach accuracy).
    phase[33:231] = "baseline"

    # The adaptation phase consisted of 110 reaches toward a
    # single target positioned at 0° in the frontal plane
    # (straight ahead; see Fig. 1b). During this phase, endpoint
    # feedback was rotated about the starting position by 30°
    # (CW or CCW; counterbalanced between participants).
    phase[231:341] = "adaptation"

    # The generalization phase consisted of 66 reaches to 1 of
    # 11 target directions (10 untrained directions) presented
    # in pseudorandom order without visual feedback.
    phase[341:] = "generalization"

    # TODO: this needs to reflect the actual number of sessions
    phase = np.concatenate([phase, phase])

    d_trl["phase"] = phase

    d_trl["su"] = d_trl["su"].astype("category")

    d_trl["ep"] = (d_trl["ep"] * 180 / np.pi) + 90
    d_trl["rotation"] = d_trl["rotation"] * 180 / np.pi

    d = pd.merge(d_mv,
                 d_trl,
                 how="outer",
                 on=["condition", "subject", "session", "trial"])

    d = d.groupby(["condition", "subject", "session", "trial"],
                  group_keys=False).apply(compute_kinematics)

    dd = d.groupby(
        ["condition", "subject", "session", "phase", "target_angle", "trial"],
        group_keys=False).apply(interpolate_movements)

    ddd = (dd.groupby([
        "condition", "subject", "session", "phase", "target_angle", "relsamp"
    ])[["t", "x", "y", "v"]].mean().reset_index())

    if plot_sub_trajectories:

        lbx, ubx = -200, 200
        lby, uby = -200, 200

        fig, ax = plt.subplots(2, 2, squeeze=False)
        ax = ax.flatten()
        for i, p in enumerate(d.phase.unique()):
            sns.scatterplot(data=d[d["phase"] == p],
                            x="x",
                            y="y",
                            hue="trial",
                            legend=False,
                            ax=ax[i])
            ax[i].set_xlim(lbx, ubx)
            ax[i].set_ylim(lby, uby)
            ax[i].set_title("Phase: " + p)
            plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(2, 2, squeeze=False)
        ax = ax.flatten()
        for i, p in enumerate(dd.phase.unique()):
            sns.scatterplot(data=dd[dd["phase"] == p],
                            x="x",
                            y="y",
                            hue="trial",
                            legend=False,
                            ax=ax[i])
            ax[i].set_xlim(lbx, ubx)
            ax[i].set_ylim(lby, uby)
            ax[i].set_title("Phase: " + p)
            plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(2, 2, squeeze=False)
        ax = ax.flatten()
        for i, p in enumerate(ddd.phase.unique()):
            sns.scatterplot(data=ddd[ddd["phase"] == p],
                            x="x",
                            y="y",
                            hue="target_angle",
                            legend=True,
                            ax=ax[i])
            ax[i].set_xlim(lbx, ubx)
            ax[i].set_ylim(lby, uby)
            ax[i].set_title("Phase: " + p)
            plt.tight_layout()
        plt.show()

    d_rec.append(d)
    dd_rec.append(dd)
    ddd_rec.append(ddd)

d = pd.concat(d_rec)
dd = pd.concat(dd_rec)
ddd = pd.concat(ddd_rec)

d = d.reset_index(drop=True)
dd = dd.reset_index(drop=True)
ddd = ddd.reset_index(drop=True)

d.groupby(["phase"])["rotation"].unique()

d.groupby(["condition"])["subject"].unique()

dp = d.groupby(
    ["condition", "session", "phase", "trial", "target_angle", "rotation"],
    observed=True)[["imv", "emv"]].mean().reset_index()

dp.sort_values(["condition", "session", "trial"], inplace=True)

dp.loc[dp["imv"] > 160, "imv"] -= 360

dp["imv"] = dp["imv"] - dp["target_angle"]

dp["target_angle"] = dp["target_angle"].astype("category")

# NOTE: main results figure
fig, ax = plt.subplots(2, 3, squeeze=False)

# initial movement vector across trials
sns.scatterplot(
    data=dp[dp["session"] == 1],
    x="trial",
    y="imv",
    hue="target_angle",
    markers=True,
    legend=False,
    ax=ax[0, 0],
)
sns.scatterplot(
    data=dp[dp["session"] == 2],
    x="trial",
    y="imv",
    hue="target_angle",
    markers=True,
    legend=False,
    ax=ax[1, 0],
)
ax[0, 0].plot(dp[dp["session"] == 1].trial,
              dp[dp["session"] == 1].rotation * 2,
              color="black",
              linestyle="-")
ax[1, 0].plot(dp[dp["session"] == 2].trial,
              dp[dp["session"] == 2].rotation * 2,
              color="black",
              linestyle="-")
ax[0, 0].set_ylim(-100, 100)
ax[1, 0].set_ylim(-100, 100)
ax[0, 0].set_title("Session 1")
ax[1, 0].set_title("Session 2")

# generalisation function
sns.barplot(data=dp[(dp["phase"] == "generalization") & (dp["session"] == 1)],
            x="target_angle",
            y="imv",
            hue="target_angle",
            legend=False,
            ax=ax[0, 1])
sns.barplot(data=dp[(dp["phase"] == "generalization") & (dp["session"] == 2)],
            x="target_angle",
            y="imv",
            hue="target_angle",
            legend=False,
            ax=ax[1, 1])
ax[0, 1].set_ylim(-150, 150)
ax[1, 1].set_ylim(-150, 150)
for ax_ in [ax[0, 1], ax[1, 1]]:
    for label in ax_.get_xticklabels():
        label.set_rotation(45)

# generalisation trajectories
dpp = ddd[ddd["phase"] == "generalization"].groupby(
    ["condition", "session", "phase", "target_angle",
     "relsamp"])[["x", "y"]].mean().reset_index()
dpp["target_angle"] = dpp["target_angle"].astype("category")
sns.scatterplot(
    data=dpp[dpp["session"] == 1],
    x="x",
    y="y",
    hue="target_angle",
    markers=True,
    ax=ax[0, 2],
)
sns.scatterplot(
    data=dpp[dpp["session"] == 2],
    x="x",
    y="y",
    hue="target_angle",
    markers=True,
    ax=ax[1, 2],
)
for i in dpp.target_angle.unique():
    i = i + 90
    ax[0, 2].plot([0, 200 * np.cos(i * np.pi / 180)],
                  [0, 200 * np.sin(i * np.pi / 180)],
                  color="black",
                  linestyle="--")
    ax[1, 2].plot([0, 200 * np.cos(i * np.pi / 180)],
                  [0, 200 * np.sin(i * np.pi / 180)],
                  color="black",
                  linestyle="--")
ax[0, 2].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax[1, 2].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
