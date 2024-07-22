from imports import *


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
    dd["su"] = d["su"].unique()[0]

    return dd


def compute_kinematics(d):
    t = d["t"].to_numpy()
    x = d["x"].to_numpy()
    y = d["y"].to_numpy()

    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    v = np.sqrt(vx**2 + vy**2)
    d["v"] = v

    v_peak = v.max()
    ts = t[v > (0.05 * v_peak)][0]

    radius = np.sqrt(x**2 + y**2)
    theta = (np.arctan2(y, x)) * 180 / np.pi

    imv = theta[(t >= ts) & (t <= ts + 0.1)].mean()
    emv = theta[-1]

    d["imv"] = imv
    d["emv"] = emv

    return d


def plot_mpep(d):
    fig, ax = plt.subplots(2, 2, squeeze=False)
    sns.lineplot(
        data=d[d["condition"] == "slow"],
        x="trial",
        y="imv",
        hue="su",
#        style="phase",
        markers=True,
        legend=False,
        ax=ax[0, 0],
    )
    sns.lineplot(
        data=d[d["condition"] == "slow"],
        x="trial",
        y="emv",
        hue="su",
#        style="phase",
        markers=True,
        legend=False,
        ax=ax[0, 1],
    )
    sns.lineplot(
        data=d[d["condition"] == "fast"],
        x="trial",
        y="imv",
        hue="su",
#        style="phase",
        markers=True,
        legend=False,
        ax=ax[1, 0],
    )
    sns.lineplot(
        data=d[d["condition"] == "fast"],
        x="trial",
        y="emv",
        hue="su",
#        style="phase",
        markers=True,
        legend=False,
        ax=ax[1, 1],
    )
    [x.set_xlabel("Trial") for x in ax.flatten()]
    [x.set_ylabel("Initial Movement Vector") for x in ax[:, 0].flatten()]
    [x.set_ylabel("Endpoint Movement Vector") for x in ax[:, 1].flatten()]
    plt.tight_layout()
    plt.show()
