import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['figure.figsize'] = [6.0, 4.0]

from matplotlib.animation import FuncAnimation


def f(x):
    return (np.sin(x / 2.5) * 6 + x) / 2.5 - 5.5


x_test = np.linspace(0, 25, 100).reshape(-1, 1)

x_train0 = np.array([3]).reshape(-1, 1)
y_train0 = f(x_train0)
x_train1 = np.array([3, 12, 20]).reshape(-1, 1)
y_train1 = f(x_train1)

### Prior belief
gp = GPR(kernel=10 * RBF(length_scale=2.5), optimizer=None)
gp.fit(x_train0, y_train0)
prior_mean, prior_sd = gp.predict(x_test, return_std=True)
prior_mean = prior_mean.flatten()
prior_lower = prior_mean - 2 * prior_sd
prior_upper = prior_mean + 2 * prior_sd

fig, ax = plt.subplots()
line = ax.plot(x_test, prior_mean)[0]
fill = ax.fill_between(x_test.flatten(), prior_lower, prior_upper, alpha=0.3)

ax.plot(x_test, f(x_test), c='C1', linestyle='--')
ax.scatter(x_train1, y_train1, c='k', marker='x')

ax.set_xlabel('sugar in grams')
ax.set_ylabel('utility')
ax.set_ylim(-8, 8)

### Posterior belief
gp = GPR(kernel=10 * RBF(length_scale=2.5), optimizer=None)
gp.fit(x_train1, y_train1)
posterior_mean, posterior_sd = gp.predict(x_test, return_std=True)
posterior_mean = posterior_mean.flatten()
posterior_lower = posterior_mean - 2 * posterior_sd
posterior_upper = posterior_mean + 2 * posterior_sd

n_frames = 100
delta_mean = (posterior_mean - prior_mean) / n_frames
delta_lower = (posterior_lower - prior_lower) / n_frames
delta_upper = (posterior_upper - prior_upper) / n_frames


def animate(i):
    # update the mean line
    line.set_ydata(prior_mean + delta_mean  * i)

    # update CI fill
    path = fill.get_paths()[0]
    vertices = path.vertices
    vertices[1:101, 1] = prior_upper + delta_upper * i
    vertices[102:-1, 1] = (prior_lower + delta_lower * i)[::-1]


anim = FuncAnimation(fig, animate, interval=50, frames=n_frames, repeat=False)
anim.save('conditioning1.gif', writer='imagemagick')
