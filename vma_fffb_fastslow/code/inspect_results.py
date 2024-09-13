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
    t = t[:n] - t[0]
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


def add_prev(x):
    x["su_prev"] = x["su"].shift(1)
    x["delta_emv"] = np.diff(x["emv"].to_numpy(), prepend=0)
    x["delta_imv"] = np.diff(x["imv"].to_numpy(), prepend=0)
    x["fb_int"] = x["emv"] - x["imv"] 
    x["err_mp"] = x["rotation"] - x["imv"]
    x["err_mp_prev"] = x["err_mp"].shift(1)
    x["err_ep"] = x["rotation"] - x["emv"]
    x["err_ep_prev"] = x["err_ep"].shift(1)
    return x


dir_data = "../data/"

d_rec = []

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
                ax[i].set_title("subject " + str(d.subject.unique()[0]) +
                                " - " + d.condition.unique()[0] + " - Phase " +
                                str(i + 1))
                plt.tight_layout()
            plt.show()

        d_rec.append(d)

d = pd.concat(d_rec)
d = d.reset_index(drop=True)
d.groupby(["condition"])["subject"].unique()

exc_subs = [8]
d = d[~d["subject"].isin(exc_subs)]

# NOTE: begin trajectory-level analysis
dd = d.groupby(["condition", "subject", "phase", "trial", "su"],
               group_keys=False).apply(interpolate_movements)

ddd = (dd.groupby(["condition", "phase", "su",
                   "relsamp"])[["t", "x", "y", "v"]].mean().reset_index())

# NOTE: trajectory by phase
fig, ax = plt.subplots(4, 2, squeeze=False, figsize=(7, 10))
fig.subplots_adjust(hspace=0.6, wspace=0.4)
for i, p in enumerate(ddd.phase.unique()):
    sns.scatterplot(data=ddd[(ddd["phase"] == p)
                             & (ddd["condition"] == "slow")],
                    x="x",
                    y="y",
                    hue="su",
                    ax=ax[i, 0])
    sns.scatterplot(data=ddd[(ddd["phase"] == p)
                             & (ddd["condition"] == "fast")],
                    x="x",
                    y="y",
                    hue="su",
                    ax=ax[i, 1])
    ax[i, 0].set_title("Condition: Slow " + "Phase: " + str(i + 1))
    ax[i, 1].set_title("Condition: Fast " + "Phase: " + str(i + 1))
for i, x in enumerate(ax.flatten()):
    x.set_xlim(-20, 20)
    x.set_ylim(0, 200)
    x.legend(loc="upper left", bbox_to_anchor=(0.0, 1), ncol=1)
plt.savefig("../figures/fig_traj_by_phase.png")

# NOTE: velocity by phase
fig, ax = plt.subplots(4, 2, squeeze=False, figsize=(7, 10))
fig.subplots_adjust(hspace=0.6, wspace=0.4)
for i, p in enumerate(ddd.phase.unique()):
    sns.lineplot(data=ddd[(ddd["phase"] == p)
                          & (ddd["condition"] == "slow")],
                 x="t",
                 y="v",
                 hue="su",
                 ax=ax[i, 0])
    sns.lineplot(data=ddd[(ddd["phase"] == p)
                          & (ddd["condition"] == "fast")],
                 x="t",
                 y="v",
                 hue="su",
                 ax=ax[i, 1])
    ax[i, 0].set_title("Condition: Slow " + "Phase: " + str(i + 1))
    ax[i, 1].set_title("Condition: Fast " + "Phase: " + str(i + 1))
for i, x in enumerate(ax.flatten()):
    x.legend(loc="upper left", bbox_to_anchor=(0.0, 1), ncol=1)
plt.savefig("../figures/fig_vel_by_phase.png")

# NOTE: Begin trial-level analysis
dp = d[[
    "condition", "subject", "trial", "phase", "su", "imv", "emv", "rotation"
]].drop_duplicates()

dp = dp.groupby(["condition", "subject"], group_keys=False).apply(add_prev)

dp["su"] = dp["su"].cat.reorder_categories(["low", "mid", "high", "inf"],
                                           ordered=True)
dp["su_prev"] = dp["su_prev"].cat.reorder_categories(
    ["low", "mid", "high", "inf"], ordered=True)

dpp = dp.groupby(["condition", "trial", "phase", "su_prev"],
                 observed=True)[["imv", "emv", "err_ep",
                                 "rotation"]].mean().reset_index()

dpp.to_csv("../data_summary/adapt_by_trial_dpp.csv", index=False)

# NOTE: fig adapt by trial
fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(12, 8))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
[
    sns.lineplot(data=dpp,
                 x="trial",
                 y="rotation",
                 color=(0.2, 0.2, 0.2),
                 legend=False,
                 ax=x) for x in ax.flatten()
]
for i, condition in enumerate(["slow", "fast"]):
    for j, su in enumerate(["low", "mid", "high", "inf"]):
        dppp = dpp[(dpp["condition"] == condition) & (dpp["su_prev"] == su)]
        sns.scatterplot(data=dppp,
                        x="trial",
                        y="imv",
                        palette=sns.color_palette("Set1")[j],
                        label=su,
                        ax=ax[0, i])
        sns.scatterplot(data=dppp,
                        x="trial",
                        y="emv",
                        palette=sns.color_palette("Set1")[j],
                        label=su,
                        ax=ax[1, i])
[x.set_ylim(-15, 30) for x in ax.flatten()]
[x.set_xlabel("Trial") for x in ax.flatten()]
[x.set_ylabel("Endppoint Movement Vector") for x in [ax[0, 1], ax[1, 1]]]
[x.set_ylabel("Initial Movement Vector") for x in [ax[0, 0], ax[1, 0]]]
[ax[0, 0].set_title("Fast"), ax[0, 1].set_title("Fast")]
[ax[1, 0].set_title("Slow"), ax[1, 1].set_title("Slow")]
[
    x.legend(loc="upper left", bbox_to_anchor=(0.0, 1), ncol=2)
    for x in ax.flatten()
]
for x in ax.flatten():
    legend = x.legend_
    legend.set_alpha(1)
plt.savefig("../figures/fig_adapt_by_trial.png")

# NOTE: scatter plots within- and between- movement correction
dpp_su = dp[np.isin(dp["phase"], [1, 2])].groupby(
    ["condition", "subject", "su", "trial"],
    observed=True)[["delta_imv", "fb_int", "err_mp",
                    "err_mp_prev"]].mean().reset_index()

dpp_su_prev = dp[np.isin(dp["phase"], [1, 2])].groupby(
    ["condition", "subject", "su_prev", "trial"],
    observed=True)[["delta_imv", "fb_int", "err_mp",
                    "err_mp_prev"]].mean().reset_index()

dpp_su.to_csv("../data_summary/scatter_dpp_su.csv", index=False)
dpp_su_prev.to_csv("../data_summary/scatter_dpp_su_prev.csv", index=False)

fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(12, 8))

sns.scatterplot(
    data=dpp_su_prev[dpp_su_prev["condition"] == "slow"],
    x="err_mp_prev",
    y="delta_imv",
    hue="su_prev",
    ax=ax[0, 0],
)
[
    sns.regplot(data=dpp_su_prev[(dpp_su_prev["condition"] == "slow")
                                 & (dpp_su_prev["su_prev"] == x)],
                x="err_mp_prev",
                y="delta_imv",
                scatter=False,
                ax=ax[0, 0]) for x in dpp_su_prev["su_prev"].unique()
]

sns.scatterplot(
    data=dpp_su[dpp_su["condition"] == "slow"],
    x="err_mp",
    y="fb_int",
    hue="su",
    ax=ax[0, 1],
)
[
    sns.regplot(data=dpp_su[(dpp_su["condition"] == "slow")
                            & (dpp_su["su"] == x)],
                x="err_mp",
                y="fb_int",
                scatter=False,
                ax=ax[0, 1]) for x in dpp_su["su"].unique()
]

sns.scatterplot(
    data=dpp_su_prev[dpp_su_prev["condition"] == "fast"],
    hue="su_prev",
    x="err_mp_prev",
    y="delta_imv",
    ax=ax[1, 0],
)
[
    sns.regplot(data=dpp_su_prev[(dpp_su_prev["condition"] == "fast")
                                 & (dpp_su_prev["su_prev"] == x)],
                x="err_mp_prev",
                y="delta_imv",
                scatter=False,
                ax=ax[1, 0]) for x in dpp_su_prev["su_prev"].unique()
]

sns.scatterplot(
    data=dpp_su[dpp_su["condition"] == "fast"],
    x="err_mp",
    y="fb_int",
    hue="su",
    ax=ax[1, 1],
)
[
    sns.regplot(data=dpp_su[(dpp_su["condition"] == "fast")
                            & (dpp_su["su"] == x)],
                x="err_mp",
                y="fb_int",
                scatter=False,
                ax=ax[1, 1]) for x in dpp_su["su"].unique()
]

ax[0, 0].set_title("Slow")
ax[0, 1].set_title("Slow")
ax[1, 0].set_title("Fast")
ax[1, 1].set_title("Fast")

ax[0, 0].set_xlabel("Movement Error")
ax[0, 0].set_ylabel("Delta Initial Movement Vector")
ax[1, 0].set_xlabel("Movement Error")
ax[1, 0].set_ylabel("Delta Initial Movement Vector")
ax[0, 1].set_xlabel("Movement Error")
ax[0, 1].set_ylabel("Within-Movement Correction")
ax[1, 1].set_xlabel("Movement Error")
ax[1, 1].set_ylabel("Within-Movement Correction")

plt.tight_layout()
plt.savefig("../figures/fig_scatter_within_between.png")
