import torch

from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf

import sys
sys.path.append('../')
from optimization.opt_utils import generate_init_data, fit_model, compare, \
    observe_and_append_data, get_gap

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


def single_experiment(f, dim, x_opt, noise=1e-6, budget=10, n_restarts=10,
                      n_raw_samples=20, visualize_fn=None, seed=0):
    bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])

    torch.manual_seed(seed)
    x_train, comp_train = generate_init_data(f, dim)
    gaps = torch.tensor([0]).float()

    model = fit_model(x_train, comp_train)
    if visualize_fn is not None:
        visualize_fn(model, x_train, comp_train, f)

    for i in range(budget):
        acq_fn = qNoisyExpectedImprovement(model=model, X_baseline=x_train)

        x_next, _ = optimize_acqf(
            acq_function=acq_fn, bounds=bounds, q=2, num_restarts=n_restarts,
            raw_samples=n_raw_samples
        )

        x_train, comp_train = observe_and_append_data(
            x_next, f, noise, x_train, comp_train
        )
        gaps = torch.cat([gaps, get_gap(x_train, f, x_opt)])

        _, model = fit_model(x_train, comp_train)
        if visualize_fn is not None:
            print('Query', i + 1)
            print(f'Compare {x_next[0]} against {x_next[1]}')
            print(model.covar_module.base_kernel.lengthscale.item())
            visualize_fn(model, x_train, comp_train, f)

    return x_train, comp_train, gaps


def repeated_experiments(f, dim, x_opt, n_trials=10, noise=1e-6, budget=10,
                         n_restarts=10, n_raw_samples=20):
    x_train = []
    comp_train = []
    gaps = []

    for i in tqdm(range(n_trials)):
        tmp_x_train, tmp_comp_train, tmp_gaps = single_experiment(
            f, dim, x_opt, noise=noise, budget=budget, n_restarts=n_restarts,
            n_raw_samples=n_raw_samples, seed=i
        )

        x_train.append(tmp_x_train)
        comp_train.append(tmp_comp_train)
        gaps.append(tmp_gaps)

    return torch.stack(x_train), torch.stack(comp_train), torch.stack(gaps)
