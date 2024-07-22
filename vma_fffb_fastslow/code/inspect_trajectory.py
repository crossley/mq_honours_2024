from imports import *
from util_func import *

d = load_all_data()
d = d.sort_values(["condition", "subject", "trial"])

d = d.groupby(["condition", "subject", "trial"], group_keys=False).apply(
    compute_kinematics
)

dd = d.groupby(
    ["condition", "subject", "phase", "trial", "sig_mp"], group_keys=False
).apply(interpolate_movements)

ddd = (
    dd.groupby(["condition", "subject", "phase", "sig_mp", "relsamp"])[
        ["time", "x", "y", "v"]
    ]
    .mean()
    .reset_index()
)

print(d)
print(dd)
print(ddd)

# d.plot(subplots=True)
# plt.show()
# dd.plot(subplots=True)
# plt.show()
# ddd.plot(subplots=True)
# plt.show()

plot_mpep(d)

fig, ax = plt.subplots(2, 2, squeeze=False)
sns.scatterplot(
    data=ddd[ddd["phase"] == "adapt_1"], x="x", y="y", hue="sig_mp", ax=ax[0, 0]
)
sns.scatterplot(
    data=ddd[ddd["phase"] == "adapt_2"], x="x", y="y", hue="sig_mp", ax=ax[0, 1]
)
sns.scatterplot(
    data=ddd[ddd["phase"] == "adapt_3"], x="x", y="y", hue="sig_mp", ax=ax[1, 0]
)
sns.scatterplot(
    data=ddd[ddd["phase"] == "adapt_4"], x="x", y="y", hue="sig_mp", ax=ax[1, 1]
)
ax[0, 0].set_title("adapt 1")
ax[0, 1].set_title("adapt 2")
ax[1, 0].set_title("adapt 3")
ax[1, 1].set_title("adapt 4")
plt.tight_layout()
plt.show()

# # dddd = dd.groupby(["subject", "relsamp"])[["time", "x", "y", "v"]].mean().reset_index()

# # dddd.plot(subplots=True)
# # plt.show()
