import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import CubicSpline


def interpolate_movements(d):
    t = d["t"]
    x = d["x"]
    y = d["y"]
    v = d["v"]

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


dir_data = "../data/"

d_rec = []

for s in range(1, 11):

    f_trl = "sub_{}_data.csv".format(s)
    f_mv = "sub_{}_data_move.csv".format(s)

    d_trl = pd.read_csv(os.path.join(dir_data, f_trl))
    d_mv = pd.read_csv(os.path.join(dir_data, f_mv))

    # NOTE: must have started counting differently for the
    # two data structures
    d_mv["trial"] += 1

    n_trial_trl = d_trl["trial"].nunique()
    n_trial_mv = d_mv["trial"].nunique()

    if (n_trial_trl == 300) and (n_trial_mv == 300):

        d_trl = d_trl.sort_values(["condition", "subject", "trial"])
        d_mv = d_mv.sort_values(["condition", "subject", "t", "trial"])

        d_hold = d_mv[d_mv["state"].isin(["state_holding"])]
        x_start = d_hold.x.mean()
        y_start = d_hold.y.mean()

        d_mv = d_mv[d_mv["state"].isin(
            ["state_moving", "state_feedback_mp", "state_moving_2"])]

        phase = np.zeros(n_trial_trl)
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

        d_rec.append(d)

        # NOTE: inspect trajectories
        fig, ax = plt.subplots(2, 2, squeeze=False)
        ax = ax.flatten()
        sns.scatterplot(data=d[d["phase"] == 1],
                        x="x",
                        y="y",
                        hue="trial",
                        legend=False,
                        ax=ax[0])
        sns.scatterplot(data=d[d["phase"] == 2],
                        x="x",
                        y="y",
                        hue="trial",
                        legend=False,
                        ax=ax[1])
        sns.scatterplot(data=d[d["phase"] == 3],
                        x="x",
                        y="y",
                        hue="trial",
                        legend=False,
                        ax=ax[2])
        sns.scatterplot(data=d[d["phase"] == 4],
                        x="x",
                        y="y",
                        hue="trial",
                        legend=False,
                        ax=ax[3])
        [x.set_xlim(-100, 100) for x in ax]
        [x.set_ylim(-20, 320) for x in ax]
        [x.set_title(dd.condition.unique()) for x in ax]
        plt.tight_layout()
        plt.show()

        def add_prev(x):
            x["su_prev"] = x["su"].shift(1)
            x["delta_emv"] = np.diff(x["emv"].to_numpy(), prepend=0)
            x["delta_imv"] = np.diff(x["imv"].to_numpy(), prepend=0)
            return x

        d = d.groupby(["condition", "subject"],
                      group_keys=False).apply(add_prev)

        dd = d.groupby(["condition", "subject", "phase", "trial"],
                       group_keys=False).apply(interpolate_movements)

        ddd = (dd.groupby(
            ["condition", "subject", "phase", "su_prev",
             "relsamp"])[["t", "x", "y", "v"]].mean().reset_index())

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

d = pd.concat(d_rec)

d.groupby(["condition"])["subject"].unique()

# NOTE: make sure this does what we want
# d["su"] = d["su"].bfill()


def add_prev(x):
    x["su_prev"] = x["su"].shift(1)
    x["delta_emv"] = np.diff(x["emv"].to_numpy(), prepend=0)
    x["delta_imv"] = np.diff(x["imv"].to_numpy(), prepend=0)
    return x


d = d.groupby(["condition", "subject"], group_keys=False).apply(add_prev)

dp = d.groupby(["condition", "trial", "su_prev", "rotation"],
               observed=True)[["imv", "emv", "delta_imv",
                               "delta_emv"]].mean().reset_index()

fig, ax = plt.subplots(2, 2, squeeze=False)
sns.scatterplot(
    data=dp[dp["condition"] == "fast"],
    x="trial",
    y="imv",
    hue="su_prev",
    markers=True,
    ax=ax[0, 0],
)
sns.scatterplot(
    data=dp[dp["condition"] == "fast"],
    x="trial",
    y="emv",
    hue="su_prev",
    markers=True,
    ax=ax[0, 1],
)
sns.scatterplot(
    data=dp[dp["condition"] == "slow"],
    x="trial",
    y="imv",
    hue="su_prev",
    markers=True,
    ax=ax[1, 0],
)
sns.scatterplot(
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

# NOTE: inspect trajectories
fig, ax = plt.subplots(2, 2, squeeze=False)
ax = ax.flatten()
sns.scatterplot(data=d[d["phase"] == 1],
                x="x",
                y="y",
                hue="trial",
                legend=False,
                ax=ax[0])
sns.scatterplot(data=d[d["phase"] == 2],
                x="x",
                y="y",
                hue="trial",
                legend=False,
                ax=ax[1])
sns.scatterplot(data=d[d["phase"] == 3],
                x="x",
                y="y",
                hue="trial",
                legend=False,
                ax=ax[2])
sns.scatterplot(data=d[d["phase"] == 4],
                x="x",
                y="y",
                hue="trial",
                legend=False,
                ax=ax[3])
theta = (90 - 30) * np.pi / 180
[x.axline((0, 0), (np.cos(theta), np.sin(theta)), color="red") for x in ax]
[
    x.set_title(f"emv: {d[d['phase'] == i+1]['emv'].mean():.2f}")
    for i, x in enumerate(ax)
]
# [x.set_xlim(-200, 200) for x in ax]
# [x.set_ylim(-200, 200) for x in ax]
plt.tight_layout()
plt.show()

# TODO: why are some data missing below?
dddd["delta_imv"] = dddd["imv"] - dddd["imv"].shift(1)
dddd["fb_int"] = dddd["imv"] - dddd["emv"]

fig, ax = plt.subplots(2, 2, squeeze=False)
sns.barplot(
    data=dddd[dddd["condition"] == "slow"],
    x="su_prev",
    y="delta_imv",
    ax=ax[0, 0],
)
sns.barplot(
    data=dddd[dddd["condition"] == "slow"],
    x="su_prev",
    y="fb_int",
    ax=ax[0, 1],
)
sns.barplot(
    data=dddd[dddd["condition"] == "fast"],
    x="su",
    y="delta_imv",
    ax=ax[1, 0],
)
sns.barplot(
    data=dddd[dddd["condition"] == "fast"],
    x="su",
    y="fb_int",
    ax=ax[1, 1],
)
plt.tight_layout()
plt.show()
