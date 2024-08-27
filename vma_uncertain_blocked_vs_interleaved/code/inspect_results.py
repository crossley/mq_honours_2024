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

for s in range(13, 31):

    f_trl = "sub_{}_data.csv".format(s)
    f_mv = "sub_{}_data_move.csv".format(s)

    d_trl = pd.read_csv(os.path.join(dir_data, f_trl))
    d_mv = pd.read_csv(os.path.join(dir_data, f_mv))

    d_trl = d_trl.sort_values(["condition", "subject", "trial"])
    d_mv = d_mv.sort_values(["condition", "subject", "t", "trial"])

    d_hold = d_mv[d_mv["state"].isin(["state_holding"])]
    x_start = d_hold.x.mean()
    y_start = d_hold.y.mean()

    d_mv = d_mv[d_mv["state"].isin(["state_moving"])]

    phase = np.zeros(d_trl["trial"].nunique())
    phase[:30] = 1
    phase[30:130] = 2
    phase[130:180] = 3
    phase[180:230] = 4
    phase[230:330] = 5
    phase[330:380] = 6
    phase[380:] = 7
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

d = pd.concat(d_rec)

d.groupby(["condition"])["subject"].unique()

d.sort_values(["condition", "subject", "trial", "t"], inplace=True)

# for s in d["subject"].unique():
#     fig, ax = plt.subplots(1, 1, squeeze=False)
#     ax[0, 0].plot(d[d["subject"] == s]["su"])
#     ax[0, 0].set_title(f"Subject {s}")
#     plt.show()

# high low
# 13, 15, 17, 25

# low high
# 19, 21, 23, 27, 29

d.loc[(d["condition"] == "blocked") & np.isin(d["subject"], [13, 15, 17, 25]),
      "condition"] = "Blocked - High low"
d.loc[(d["condition"] == "blocked")
      & np.isin(d["subject"], [19, 21, 23, 27, 29]),
      "condition"] = "Blocked - Low High"

d.groupby(["condition"])["subject"].unique()

# NOTE: because of bug in experiment
d["rotation"] = d["rotation"] * 2

d.groupby(["condition", "subject"])["trial"].nunique()


def add_prev(x):
    x["su_prev"] = x["su"].shift(1)
    x["delta_emv"] = np.diff(x["emv"].to_numpy(), prepend=0)
    x["movement_error"] = x["rotation"] - x["emv"]
    return x


dp = d[["condition", "subject", "trial", "phase", "su", "emv",
        "rotation"]].drop_duplicates()

dp = dp.groupby(["condition", "subject"], group_keys=False).apply(add_prev)

dpp = dp.groupby(["condition", "trial", "phase", "su_prev", "rotation"],
                 observed=True)[["emv", "delta_emv",
                                 "movement_error"]].mean().reset_index()

fig, ax = plt.subplots(3, 2, squeeze=False)

# emv
sns.scatterplot(
    data=dpp[dpp["condition"] != "interleaved"],
    x="trial",
    y="emv",
    style="condition",
    hue="su_prev",
    markers=True,
    legend="full",
    ax=ax[0, 0],
)
sns.scatterplot(
    data=dpp[dpp["condition"] == "interleaved"],
    x="trial",
    y="emv",
    hue="su_prev",
    markers=True,
    legend="full",
    ax=ax[0, 1],
)
[x.set_ylim(-10, 50) for x in ax.flatten()]
[x.set_xlabel("Trial") for x in ax.flatten()]
[x.set_ylabel("Endppoint Movement Vector") for x in ax.flatten()]
# delta emv
sns.scatterplot(
    data=dpp[dpp["condition"] != "interleaved"],
    x="trial",
    y="delta_emv",
    style="condition",
    hue="su_prev",
    markers=True,
    legend=False,
    ax=ax[1, 0],
)
sns.scatterplot(
    data=dpp[dpp["condition"] == "interleaved"],
    x="trial",
    y="delta_emv",
    hue="su_prev",
    markers=True,
    legend=False,
    ax=ax[1, 1],
)
[x.set_xlabel("Trial") for x in ax.flatten()]
[x.set_ylabel("Delta Endppoint Movement Vector") for x in ax.flatten()]

# add rotation to the above axes
[
    sns.lineplot(
        data=dpp[dpp["condition"] != "interleaved"],
        x="trial",
        y="rotation",
        hue="condition",
        palette=['k'],
        legend=False,
        ax=ax_,
    ) for ax_ in ax.flatten()[:4]
]

# add scatter plot of delta emv vs movement error colour
# coded by su_prev with reg lines
sns.scatterplot(data=dpp[(dpp["condition"] != "interleaved") & (dpp["phase"] == 2)],
                y="delta_emv",
                x="movement_error",
                hue="su_prev",
                style="condition",
                legend=False,
                ax=ax[2, 0])
sns.scatterplot(data=dpp[(dpp["condition"] == "interleaved") & (dpp["phase"] == 2)],
                y="delta_emv",
                x="movement_error",
                hue="su_prev",
                legend=False,
                ax=ax[2, 1])
sns.regplot(data=dpp[(dpp["condition"] != "interleaved")
                     & (dpp["su_prev"] == dpp["su_prev"].unique()[0]) &
                     (dpp["phase"] == 2)],
            x="delta_emv",
            y="movement_error",
            scatter=False,
            robust=True,
            ax=ax[2, 0])
sns.regplot(data=dpp[(dpp["condition"] != "interleaved")
                     & (dpp["su_prev"] == dpp["su_prev"].unique()[1]) &
                     (dpp["phase"] == 2)],
            y="delta_emv",
            x="movement_error",
            scatter=False,
            robust=True,
            ax=ax[2, 0])
sns.regplot(data=dpp[(dpp["condition"] == "interleaved")
                     & (dpp["su_prev"] == dpp["su_prev"].unique()[0]) &
                     (dpp["phase"] == 2)],
            x="delta_emv",
            y="movement_error",
            scatter=False,
            robust=True,
            ax=ax[2, 1])
sns.regplot(data=dpp[(dpp["condition"] == "interleaved")
                     & (dpp["su_prev"] == dpp["su_prev"].unique()[1]) &
                     (dpp["phase"] == 2)],
            y="delta_emv",
            x="movement_error",
            scatter=False,
            robust=True,
            ax=ax[2, 1])
[x.set_xlabel("Delta Endppoint Movement Vector") for x in ax.flatten()]
[x.set_ylabel("Movement Error") for x in ax.flatten()]
ax[0, 0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.4), ncol=2)
ax[0, 1].legend(loc="upper left", bbox_to_anchor=(0.0, 1.4), ncol=2)
plt.show()

d.to_csv("../data_summary/summary.csv")

# # NOTE: inspect trajectories
# fig, ax = plt.subplots(3, 3, squeeze=False)
# ax = ax.flatten()
# sns.scatterplot(data=d[d["phase"] == 1],
#                 x="x",
#                 y="y",
#                 hue="trial",
#                 legend=False,
#                 ax=ax[0])
# sns.scatterplot(data=d[d["phase"] == 2],
#                 x="x",
#                 y="y",
#                 hue="trial",
#                 legend=False,
#                 ax=ax[1])
# sns.scatterplot(data=d[d["phase"] == 3],
#                 x="x",
#                 y="y",
#                 hue="trial",
#                 legend=False,
#                 ax=ax[2])
# sns.scatterplot(data=d[d["phase"] == 4],
#                 x="x",
#                 y="y",
#                 hue="trial",
#                 legend=False,
#                 ax=ax[3])
# sns.scatterplot(data=d[d["phase"] == 5],
#                 x="x",
#                 y="y",
#                 hue="trial",
#                 legend=False,
#                 ax=ax[4])
# sns.scatterplot(data=d[d["phase"] == 6],
#                 x="x",
#                 y="y",
#                 hue="trial",
#                 legend=False,
#                 ax=ax[5])
# sns.scatterplot(data=d[d["phase"] == 7],
#                 x="x",
#                 y="y",
#                 hue="trial",
#                 legend=False,
#                 ax=ax[6])
# theta = (90 - 30) * np.pi / 180
# [x.axline((0, 0), (np.cos(theta), np.sin(theta)), color="red") for x in ax]
# [
#     x.set_title(f"emv: {d[d['phase'] == i+1]['emv'].mean():.2f}")
#     for i, x in enumerate(ax)
# ]
# [x.set_xlim(-200, 200) for x in ax]
# [x.set_ylim(-200, 200) for x in ax]
# plt.tight_layout()
# plt.show()

# dd = d.groupby(["condition", "subject", "phase", "trial"],
#                group_keys=False).apply(interpolate_movements)

# fig, ax = plt.subplots(3, 1, squeeze=False)
# sns.scatterplot(data=dd, x="x", y="y", hue="su", ax=ax[0, 0])
# sns.scatterplot(data=dd, x="trial", y="imv", hue="su", ax=ax[1, 0])
# sns.scatterplot(data=dd, x="trial", y="emv", hue="su", ax=ax[2, 0])
# plt.tight_layout()
# plt.show()

# ddd = (dd.groupby(["condition", "subject", "phase", "su",
#                    "relsamp"])[["t", "x", "y", "v"]].mean().reset_index())

# fig, ax = plt.subplots(1, 1, squeeze=False)
# sns.scatterplot(data=ddd, x="x", y="y", hue="su", style="phase", ax=ax[0, 0])
# plt.show()
