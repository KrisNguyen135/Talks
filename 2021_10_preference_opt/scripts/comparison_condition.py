import numpy as np
import torch
import gpytorch

import sys
sys.path.append('../')

from optimization.opt_utils import fit_model

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

x_train = torch.tensor([[3], [12], [20]]).float()
comp_train0 = torch.tensor([
    [0, 1],  # f(3) > f(12)
    [2, 1],  # f(12) < f(20)
]).long()
comp_train1 = torch.tensor([
    [0, 1],  # f(3) > f(12)
    [2, 1],  # f(12) < f(20)
    [2, 0],  # f(3) < f(20)
]).long()

### Prior belief
if comp_train0.numel() == 0:
    prior_mean = np.zeros(x_test.size)
    prior_lower = - 2 * np.ones(x_test.size)
    prior_upper = 2 * np.ones(x_test.size)
else:
    model = fit_model(
        x_train, comp_train0,
        covar_module=gpytorch.kernels.RBFKernel(
            lengthscale_constraint=gpytorch.constraints.GreaterThan(2.5)
        )
    )

    with torch.no_grad():
        output = model(torch.tensor(x_test))
        lower, upper = output.confidence_region()

    prior_mean = output.mean.detach().numpy()
    prior_lower = lower.detach().numpy()
    prior_upper = upper.detach().numpy()

fig, ax = plt.subplots()
line = ax.plot(x_test, prior_mean)[0]
fill = ax.fill_between(x_test.flatten(), prior_lower, prior_upper, alpha=0.3)

ax.plot(x_test, f(x_test) / 2.5, c='C1', linestyle='--')
# for comp_ind in range(comp_train1.shape[0]):
#     tmp_comp = comp_train1[comp_ind, :]
#
#     plt.scatter(
#         x_train[tmp_comp], f(x_train[tmp_comp]) / 2.5,
#         marker='x', c='k'
#     )
for ind in comp_train1.flatten():
    plt.scatter(x_train[ind], f(x_train[ind]) / 2.5, marker='x', c='k')

ax.set_xlabel('sugar in grams')
ax.set_ylabel('utility')
ax.set_ylim(-3, 3)

### Posterior belief
model = fit_model(
    x_train, comp_train1,
    covar_module=gpytorch.kernels.RBFKernel(
        lengthscale_constraint=gpytorch.constraints.GreaterThan(2.5)
    )
)

with torch.no_grad():
    output = model(torch.tensor(x_test))
    lower, upper = output.confidence_region()

posterior_mean = output.mean.detach().numpy()
posterior_lower = lower.detach().numpy()
posterior_upper = upper.detach().numpy()

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
anim.save('comparison_condition2.gif', writer='imagemagick')
