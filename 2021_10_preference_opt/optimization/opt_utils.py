import numpy as np
import torch

import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.constraints import Interval

from botorch.models.pairwise_gp import PairwiseGP, \
    PairwiseLaplaceMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model

from itertools import combinations

import matplotlib.pyplot as plt


def generate_init_data(f, dim, n=2):
    x = torch.rand(n, dim).float()
    y = f(x)
    comp = compare(y)

    return x, comp


def fit_model(x_train, comp_train, covar_module=None):
    # if covar_module is None:
    #     covar_module = ScaleKernel(
    #         RBFKernel(
    #             ard_num_dims=x_train.shape[-1],
    #             lengthscale_constraint=Interval(0, 0.25)
    #         )
    #     )

    model = PairwiseGP(x_train, comp_train, covar_module=covar_module)
    mll = PairwiseLaplaceMarginalLogLikelihood(model)

    # hacky way to ignore jitter warnings
    with gpytorch.settings.cholesky_jitter(1e-2):
        fit_gpytorch_model(mll)

    return model


def compare(y, n_comp=1, noise=1e-6):
    all_pairs = np.array(list(combinations(range(y.shape[0]), 2)))
    # randomly choose some pairs to make comparisons
    # (choose all pairs by default)
    comp_pairs = all_pairs[
        np.random.choice(range(len(all_pairs)), n_comp, replace=False)
    ]

    c0 = y[comp_pairs[:, 0]] + np.random.normal(size=n_comp) * noise
    c1 = y[comp_pairs[:, 1]] + np.random.normal(size=n_comp) * noise

    reverse_comp = (c0 < c1).numpy().flatten()

    comp_pairs[reverse_comp, :] = np.flip(comp_pairs[reverse_comp, :], 1)
    return torch.tensor(comp_pairs).long()


def observe_and_append_data_old(x_next, f, noise, x_train, comp_train):
    """
    x_next should be an experiment, i.e., a pair of locations
    """
    x_next = x_next.to(x_train)
    y_next = f(x_next)
    comp_next = compare(y_next, noise=noise)
    comp_train = torch.cat([comp_train, comp_next + x_train.shape[-2]])
    x_train = torch.cat([x_train, x_next])

    return x_train, comp_train


def observe_and_append_data(x_next, f, noise, x_train, comp_train, tol=1e-3):
    """
    x_next should be an experiment, i.e., a pair of locations
    """
    x_next = x_next.to(x_train)
    y_next = f(x_next)
    comp_next = compare(y_next, noise=noise)

    n = x_train.shape[-2]
    new_x_train = x_train.clone()
    new_comp_next = comp_next.clone() + n
    n_dups = 0

    ### first element
    dup_ind = torch.where(
        torch.all(
            torch.isclose(x_train, x_next[0], atol=tol),
            axis=1
        )
    )[0]
    if dup_ind.nelement() == 0:
        new_x_train = torch.cat([x_train, x_next[0].unsqueeze(-2)])
    else:
        # replace n with the duplicated index
        # decrement the other index
        new_comp_next = torch.where(
            new_comp_next == n,
            dup_ind,
            new_comp_next - 1
        )

        n_dups += 1

    ### second element
    dup_ind = torch.where(
        torch.all(
            torch.isclose(new_x_train, x_next[1], atol=tol),
            axis=1
        )
    )[0]
    if dup_ind.nelement() == 0:
        new_x_train = torch.cat([new_x_train, x_next[1].unsqueeze(-2)])
    else:
        # replace n + 1 with the duplicated index
        new_comp_next = torch.where(
            new_comp_next == n + 1 - n_dups,
            dup_ind,
            new_comp_next
        )

    new_comp_train = torch.cat([comp_train, new_comp_next])

    return new_x_train, new_comp_train


def get_gap(x_train, f, x_opt):
    f_first = f(x_train[:2]).max()
    # TODO: replace this with f(argmax posterior mean)
    f_best = f(x_train).max()
    f_opt = f(x_opt)

    return (f_best - f_first) / (f_opt - f_first)


def plot_gaps(gaps, names):
    assert len(gaps.shape) == 3  # acq funcs x # trials x time
    assert gaps.shape[0] == len(names)

    n_trials = gaps.shape[1]
    time = gaps.shape[2]
    xs = np.arange(time)

    plt.axhline(1, c='r', linestyle='--')

    for i, name in enumerate(names):
        tmp_gaps = gaps[i]

        mean_gaps = tmp_gaps.mean(axis=0)
        err_gaps = 2 * tmp_gaps.std(axis=0) / np.sqrt(n_trials)

        plt.plot(xs, mean_gaps, label=name)
        plt.fill_between(
            xs, mean_gaps + err_gaps, mean_gaps - err_gaps, alpha=0.1
        )

    plt.legend()
