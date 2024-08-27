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

    # NOTE: Deal with too_slow trials. Should have been
    # marked in the experiment code. Will fix in next
    # version.
    condition = d["condition"].unique()[0]
    if condition == "slow":
        mt_too_slow = 3500
        mt_too_fast = 2000

    elif condition == "fast":
        mt_too_slow = 2000
        mt_too_fast = 500

    n = np.where((t - t[0]) <= mt_too_slow)[0][-1]
    t = t[:n]
    x = x[:n]
    y = y[:n]
    v = v[:n]

    xs = CubicSpline(t, x)
    ys = CubicSpline(t, y)
    vs = CubicSpline(t, v)

    tt = np.linspace(t.min(), t.max(), 100)
    xx = xs(tt)
    yy = ys(tt)
    vv = vs(tt)

    relsamp = np.arange(0, tt.shape[0], 1)

    #     fig, ax = plt.subplots(1, 3, squeeze=False, figsize=(15, 5))
    #     ax = ax.flatten()
    #     sns.scatterplot(x=x, y=y, hue=t, ax=ax[0])
    #     sns.scatterplot(x=xx, y=yy, hue=tt, ax=ax[1])
    #     sns.histplot(data=t, ax=ax[2])
    #     plt.show()

    dd = pd.DataFrame({"relsamp": relsamp, "t": tt, "x": xx, "y": yy, "v": vv})
    dd["condition"] = d["condition"].unique()[0]
    dd["subject"] = d["subject"].unique()[0]
    dd["trial"] = d["trial"].unique()[0]
    dd["phase"] = d["phase"].unique()[0]
    dd["su"] = d["su"].unique()[0]
    dd["su_prev"] = d["su_prev"].unique()[0]
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


# TODO: give this function a better name
def add_prev(x):
    x["su_prev"] = x["su"].shift(1)
    x["delta_emv"] = np.diff(x["emv"].to_numpy(), prepend=0)
    x["delta_imv"] = np.diff(x["imv"].to_numpy(), prepend=0)
    x["fb_int"] = x["imv"] - x["emv"]
    return x


dir_data = "../data/"

d_rec = []
dd_rec = []
ddd_rec = []

plot_sub_trajectories = False

for s in range(1, 16):

    f_trl = "sub_{}_data.csv".format(s)
    f_mv = "sub_{}_data_move.csv".format(s)

    try:
        d_trl = pd.read_csv(os.path.join(dir_data, f_trl))
        d_mv = pd.read_csv(os.path.join(dir_data, f_mv))

        # NOTE: must have started counting differently for the
        # two data structures
        d_mv["trial"] += 1

        n_trial_trl = d_trl["trial"].nunique()
        n_trial_mv = d_mv["trial"].nunique()

    except FileNotFoundError:
        n_trial_trl = 0
        n_trial_mv = 0

    if (n_trial_trl == 300) and (n_trial_mv == 300):

        d_trl = d_trl.sort_values(["condition", "subject", "trial"])
        d_mv = d_mv.sort_values(["condition", "subject", "t", "trial"])

        d_hold = d_mv[d_mv["state"].isin(["state_holding"])]
        x_start = d_hold.x.mean()
        y_start = d_hold.y.mean()

        d_mv = d_mv[d_mv["state"].isin(
            ["state_moving", "state_feedback_mp", "state_moving_2"])]

        phase = np.zeros(d_trl.shape[0])
        phase[:75] = 1
        phase[75:150] = 2
        phase[150:225] = 3
        phase[225:] = 4
        d_trl["phase"] = phase

        d_trl["su"] = d_trl["su"].astype("category")

        d_trl["ep"] = (d_trl["ep"] * 180 / np.pi) + 90
        d_trl["rotation"] = d_trl["rotation"] * 180 / np.pi

        d = pd.merge(d_mv,
                     d_trl,
                     how="outer",
                     on=["condition", "subject", "trial"])

        d = d.groupby(["condition", "subject", "trial"],
                      group_keys=False).apply(compute_kinematics)

        d = d.groupby(["condition", "subject"],
                      group_keys=False).apply(add_prev)

        dd = d.groupby(["condition", "subject", "phase", "trial"],
                       group_keys=False).apply(interpolate_movements)

        ddd = (dd.groupby(
            ["condition", "subject", "phase", "su_prev",
             "relsamp"])[["t", "x", "y", "v"]].mean().reset_index())

        ddd["su_prev"] = ddd["su_prev"].astype("category")
        ddd["su_prev"] = ddd["su_prev"].cat.reorder_categories(
            ["low", "mid", "high", "inf"], ordered=True)

        if plot_sub_trajectories:

            lbx, ubx = -150, 150
            lby, uby = -100, 400

            fig, ax = plt.subplots(2, 2, squeeze=False)
            ax = ax.flatten()
            for i in range(0, 4):
                sns.scatterplot(data=d[d["phase"] == i + 1],
                                x="x",
                                y="y",
                                hue="trial",
                                legend=False,
                                ax=ax[i])
                ax[i].set_xlim(lbx, ubx)
                ax[i].set_ylim(lby, uby)
                ax[i].set_title("subject " +
                                str(d.subject.unique()[0]) + " - " +
                                d.condition.unique()[0] + " - Phase " + str(i + 1))
                plt.tight_layout()
            plt.show()

            fig, ax = plt.subplots(2, 2, squeeze=False)
            ax = ax.flatten()
            for i in range(0, 4):
                sns.scatterplot(data=dd[dd["phase"] == i + 1],
                                x="x",
                                y="y",
                                hue="trial",
                                legend=False,
                                ax=ax[i])
                ax[i].set_xlim(lbx, ubx)
                ax[i].set_ylim(lby, uby)
                ax[i].set_title(dd.condition.unique()[0] + " - Phase " + str(i + 1))
                plt.tight_layout()
            plt.show()

            fig, ax = plt.subplots(2, 2, squeeze=False)
            ax = ax.flatten()
            sns.scatterplot(data=ddd[ddd["phase"] == 1],
                            x="x",
                            y="y",
                            hue="su_prev",
                            ax=ax[0])
            sns.scatterplot(data=ddd[ddd["phase"] == 2],
                            x="x",
                            y="y",
                            hue="su_prev",
                            ax=ax[1])
            sns.scatterplot(data=ddd[ddd["phase"] == 3],
                            x="x",
                            y="y",
                            hue="su_prev",
                            ax=ax[2])
            sns.scatterplot(data=ddd[ddd["phase"] == 4],
                            x="x",
                            y="y",
                            hue="su_prev",
                            ax=ax[3])
            [x.set_xlim(-100, 100) for x in ax]
            [x.set_ylim(-20, 320) for x in ax]
            [x.set_title(ddd.condition.unique()) for x in ax]
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

d["su"] = d["su"].cat.reorder_categories(
    ["low", "mid", "high", "inf"], ordered=True)

d["su_prev"] = d["su_prev"].cat.reorder_categories(
    ["low", "mid", "high", "inf"], ordered=True)

ddd["su_prev"] = ddd["su_prev"].astype("category")
ddd["su_prev"] = ddd["su_prev"].cat.reorder_categories(
    ["low", "mid", "high", "inf"], ordered=True)

d.groupby(["condition"])["subject"].unique()

# NOTE: exlcude subjects with excessively variable
# trajectories as identified by visual inspection
exc_subs = [8]

d = d[~d["subject"].isin(exc_subs)]
dd = dd[~dd["subject"].isin(exc_subs)]
ddd = ddd[~ddd["subject"].isin(exc_subs)]

dp = d.groupby(["condition", "trial", "su_prev", "rotation"],
               observed=True)[["imv", "emv", "delta_imv",
                               "delta_emv"]].mean().reset_index()

fig, ax = plt.subplots(2, 2, squeeze=False)
sns.lineplot(
    data=dp[dp["condition"] == "fast"],
    x="trial",
    y="imv",
    hue="su_prev",
    markers=True,
    ax=ax[0, 0],
)
sns.lineplot(
    data=dp[dp["condition"] == "fast"],
    x="trial",
    y="emv",
    hue="su_prev",
    markers=True,
    ax=ax[0, 1],
)
sns.lineplot(
    data=dp[dp["condition"] == "slow"],
    x="trial",
    y="imv",
    hue="su_prev",
    markers=True,
    ax=ax[1, 0],
)
sns.lineplot(
    data=dp[dp["condition"] == "slow"],
    x="trial",
    y="emv",
    hue="su_prev",
    markers=True,
    ax=ax[1, 1],
)
sns.lineplot(
    data=dp[dp["condition"] == "fast"],
    x="trial",
    y="rotation",
    legend=False,
    ax=ax[0, 0],
)
sns.lineplot(
    data=dp[dp["condition"] == "slow"],
    x="trial",
    y="rotation",
    legend=False,
    ax=ax[0, 1],
)
sns.lineplot(
    data=dp[dp["condition"] == "fast"],
    x="trial",
    y="rotation",
    legend=False,
    ax=ax[1, 0],
)
sns.lineplot(
    data=dp[dp["condition"] == "slow"],
    x="trial",
    y="rotation",
    legend=False,
    ax=ax[1, 1],
)
[x.set_ylim(-15, 30) for x in ax.flatten()]
[x.set_xlabel("Trial") for x in ax.flatten()]
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(2, 2, squeeze=False)
sns.barplot(
    data=d[d["condition"] == "slow"],
    x="su_prev",
    y="delta_imv",
    ax=ax[0, 0],
)
sns.barplot(
    data=d[d["condition"] == "slow"],
    x="su",
    y="fb_int",
    ax=ax[0, 1],
)
sns.barplot(
    data=d[d["condition"] == "fast"],
    x="su_prev",
    y="delta_imv",
    ax=ax[1, 0],
)
sns.barplot(
    data=d[d["condition"] == "fast"],
    x="su",
    y="fb_int",
    ax=ax[1, 1],
)
ax[0, 0].set_title("Slow")
ax[0, 1].set_title("Slow")
ax[1, 0].set_title("Fast")
ax[1, 1].set_title("Fast")
plt.tight_layout()
plt.show()
