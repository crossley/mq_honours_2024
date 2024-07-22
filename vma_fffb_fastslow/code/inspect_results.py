from imports import *
from util_func import *

d_trl = pd.read_csv("../data/sub_1_data.csv")
d_mv = pd.read_csv("../data/sub_1_data_move.csv")

d_trl = d_trl.sort_values(["trial"])
d_mv = d_mv.sort_values(["t", "trial"])

d = pd.merge(d_mv, d_trl, how="outer", on=["condition", "subject", "trial"])

d = d[d["state"] == "state_moving"]

d = d.groupby(["condition", "subject", "trial"],
              group_keys=False).apply(compute_kinematics)

dd = d.groupby(["condition", "subject", "trial", "su"],
               group_keys=False).apply(interpolate_movements)

ddd = (
    dd.groupby(["condition", "subject", "su", "relsamp"])[
        ["t", "x", "y", "v"]
    ]
    .mean()
    .reset_index()
)

d.plot(subplots=True)
plt.show()
dd.plot(subplots=True)
plt.show()
ddd.plot(subplots=True)
plt.show()

plot_mpep(d)

# fig, ax = plt.subplots(2, 2, squeeze=False)
# sns.scatterplot(
#     data=ddd[ddd["phase"] == "adapt_1"], x="x", y="y", hue="su", ax=ax[0, 0]
# )
# sns.scatterplot(
#     data=ddd[ddd["phase"] == "adapt_2"], x="x", y="y", hue="su", ax=ax[0, 1]
# )
# sns.scatterplot(
#     data=ddd[ddd["phase"] == "adapt_3"], x="x", y="y", hue="su", ax=ax[1, 0]
# )
# sns.scatterplot(
#     data=ddd[ddd["phase"] == "adapt_4"], x="x", y="y", hue="su", ax=ax[1, 1]
# )
# ax[0, 0].set_title("adapt 1")
# ax[0, 1].set_title("adapt 2")
# ax[1, 0].set_title("adapt 3")
# ax[1, 1].set_title("adapt 4")
# plt.tight_layout()
# plt.show()
#
# # # dddd = dd.groupby(["subject", "relsamp"])[["time", "x", "y", "v"]].mean().reset_index()
#
# # # dddd.plot(subplots=True)
# # # plt.show()
