import numpy as np
import matplotlib.pyplot as plt


def obj_func_kalman(params, *args):
    # NOTE: Are the next two lines of code unnecessary?
    rot = args[0]
    args = rot
    x_pred = simulate_kalman(params, args)
    sse = np.sum((x_obs) ** 2)
    return sse


def simulate_kalman(params, args):
    class gaussian:
        def __init__(self, mean, variance):
            self.mean = mean
            self.variance = variance

    def update(prior, measurement):
        x, P = prior.mean, prior.variance  # mean and variance of prior
        z, R = (
            measurement.mean,
            measurement.variance,
        )  # mean and variance of measurement

        y = z - x  # residual
        K = P / (P + R)  # Kalman gain

        x = x + K * y  # posterior
        P = (1 - K) * P  # posterior variance
        return gaussian(x, P)

    def predict(posterior, measurement):
        x, P = posterior.mean, posterior.variance  # mean and variance of posterior
        dx, Q = (
            measurement.mean,
            measurement.variance,
        )  # mean and variance of measurement
        x = x + dx
        P = P + Q
        return gaussian(x, P)

    Q = params[0]  # process noise
    R = params[1]  # measurement noise
    P = params[2]  # state noise
    state_mean = params[3]  # initial state mean

    x = gaussian(state_mean, P)
    process_model = gaussian(0.0, Q)

    zs = args

    x_pred = []
    xs = []
    priors = []
    for i, z in enumerate(zs):
        prior = predict(x, process_model)
        observation_model = gaussian(z, R)
        x = update(prior, observation_model)
        priors.append(prior)
        xs.append(x)

    return xs

# TODO: https://github.com/pykalman/pykalman/blob/master/examples/standard/plot_sin.py
# TODO: https://github.com/pykalman/pykalman/blob/master/examples/standard/plot_em.py

# NOTE: R is measurement noise which is more or less the
# visual blur of each trial's midpoint feedback. But it also
# seems like it could be motor output noise based on the
# equations in the documentation.
R = 5.0

# NOTE: P is state noise which is essentially the variance
# of the perturbation schedule we imposed in our
# experiments.
P = 4.0

# NOTE: Q is process noise -- process as in the process that
# governs how our state estimates transition from one value
# to the next across trials
Q = 0.01

# initial state mean
state_mean = 0.0

# package up params
params = [Q, R, P, state_mean]

# package up args
rot = np.concatenate(
    (np.zeros(100), np.random.normal(30, P**0.5, 400), np.zeros(100))
)
args = rot

# simulate the kalman filter
xs = simulate_kalman(params, args)

fig, ax = plt.subplots(2, 1, squeeze=False)
ax[0, 0].plot(rot)
ax[1, 0].plot([x.mean for x in xs])
ax[1, 0].plot([x.mean + np.sqrt(x.variance) for x in xs], "--k")
ax[1, 0].plot([x.mean - np.sqrt(x.variance) for x in xs], "--k")
plt.show()
