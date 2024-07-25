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
    r = np.sqrt(x**2 + y**2)

    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    v = np.sqrt(vx**2 + vy**2)
    d["v"] = v

    v_peak = v.max()
    # ts = t[v > (0.05 * v_peak)][0]
    ts = t[r > 0.1 * r.max()][0]

    radius = np.sqrt(x**2 + y**2)
    theta = (np.arctan2(y, x)) * 180 / np.pi

    imv = theta[(t >= ts) & (t <= ts + 0.1)].mean()
    emv = theta[-1]

    d["imv"] = imv
    d["emv"] = emv

    return d


d_trl = pd.read_csv("../data/sub_3_data.csv")
d_mv = pd.read_csv("../data/sub_3_data_move.csv")

d_trl = d_trl.sort_values(["trial"])
d_mv = d_mv.sort_values(["t", "trial"])

# TODO: Add phase to d_trl befor emerge
phase = np.zeros(d["trial"].nunique())
phase[:30] = 1
phase[30:130] = 2
phase[130:180] = 3
phase[180:230] = 4
phase[230:330] = 5
phase[330:380] = 6
phase[380:] = 7
d_trl["phase"] = phase

d = pd.merge(d_mv, d_trl, how="outer", on=["condition", "subject", "trial"])

d = d[d["state"].isin(["state_moving"])]

d["x"] = d["x"] - d["x"].values[0]
d["y"] = d["y"] - d["y"].values[0]
d["y"] = -d["y"]

d = d.groupby(["condition", "subject", "trial"],
              group_keys=False).apply(compute_kinematics)

fig, ax = plt.subplots(1, 3, squeeze=False)
sns.scatterplot(data=d, x="x", y="y", hue="su", ax=ax[0, 0])
sns.scatterplot(data=d, x="trial", y="imv", hue="su", ax=ax[0, 1])
sns.scatterplot(data=d, x="trial", y="emv", hue="su", ax=ax[0, 2])
plt.tight_layout()
plt.show()

dd = d.groupby(["condition", "subject", "phase", "trial", "su"],
               group_keys=False).apply(interpolate_movements)

fig, ax = plt.subplots(1, 3, squeeze=False)
sns.scatterplot(data=dd, x="x", y="y", hue="su", ax=ax[0, 0])
sns.scatterplot(data=dd, x="trial", y="imv", hue="su", ax=ax[0, 1])
sns.scatterplot(data=dd, x="trial", y="emv", hue="su", ax=ax[0, 2])
plt.tight_layout()
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
