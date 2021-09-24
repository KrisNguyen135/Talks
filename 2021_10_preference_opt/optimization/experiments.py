import torch

from optimization.opt_utils import (
    generate_init_data,
    fit_model,
    compare,
    observe_and_append_data,
    get_gap,
)

from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")


def single_experiment(
    f,
    dim,
    x_opt,
    bounds,
    acq_fn,
    covar_module=None,
    noise=1e-6,
    budget=10,
    visualize_fn=None,
    progress_bar=False,
    seed=0,
):
    if visualize_fn is not None:
        progress_bar = False

    torch.manual_seed(seed)
    x_train, comp_train = generate_init_data(
        f,
        dim,
        scale=bounds[
            :,
        ],
    )
    gaps = torch.tensor([0]).float()

    model = fit_model(x_train, comp_train, covar_module=covar_module)
    if visualize_fn is not None:
        visualize_fn(model, x_train, comp_train, f)

    for i in tqdm(range(budget)) if progress_bar else range(budget):
        x_next = acq_fn.run(model, x_train)

        x_train, comp_train = observe_and_append_data(
            x_next, f, noise, x_train, comp_train
        )
        gaps = torch.cat([gaps, get_gap(x_train, f, x_opt)])

        model = fit_model(x_train, comp_train, covar_module=covar_module)
        if visualize_fn is not None:
            # print("Query", i + 1)
            # print(f"Compare {x_next[0]} against {x_next[1]}")
            # print(model.covar_module.base_kernel.lengthscale.detach().numpy())
            visualize_fn(model, x_train, comp_train, f)

    return x_train, comp_train, gaps


def repeated_experiments(
    fs,
    xs_opt,
    bounds,
    dim,
    acq_funcs,
    n_trials=10,
    noise=1e-6,
    budget=10,
    verbose=False,
):
    gaps = torch.zeros((len(fs), len(acq_funcs), n_trials, budget + 1))

    for f_ind, (f, x_opt, tmp_bounds) in enumerate(zip(fs, xs_opt, bounds)):
        if verbose:
            print("Function", f.__name__)

        for acq_ind, acq_fn in enumerate(acq_funcs):
            if verbose:
                print("\tAcq.", acq_fn.__name__)
                print("\t\t", end=" ")

            for i in range(n_trials):
                if verbose:
                    print(i, end=" ")

                acq_fn_instance = acq_fn(tmp_bounds)

                _, _, tmp_gaps = single_experiment(
                    f,
                    dim,
                    x_opt,
                    tmp_bounds,
                    acq_fn_instance,
                    noise=noise,
                    budget=budget,
                    seed=i,
                )

                gaps[f_ind, acq_ind, i, :] = tmp_gaps

            if verbose:
                print()

    return gaps
