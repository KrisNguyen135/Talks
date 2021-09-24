from policies.base import BasePolicy

from contextlib import ExitStack
import gpytorch.settings as gpts

import math
import torch
from torch.distributions.normal import Normal
from torch.quasirandom import SobolEngine
import gpytorch

from botorch.generation import MaxPosteriorSampling
from botorch.optim import optimize_acqf


class DuelingThompsonSampling(BasePolicy):
    def __init__(self, bounds, n_restarts=10, n_raw_samples=10):
        super().__init__(bounds, n_restarts=n_restarts, n_raw_samples=n_raw_samples)
        # self.x_test = x_test

        # process grid
        self.dim = len(bounds[0])
        assert self.dim == len(bounds[1])

        if self.dim == 1:
            x_test = torch.linspace(bounds[0, 0], bounds[1, 0], 101).float()
            x_test = x_test.reshape(-1, 1)
        elif self.dim == 2:
            x_lin1 = torch.linspace(bounds[0, 0], bounds[1, 0], 51).float()
            x_lin2 = torch.linspace(bounds[0, 1], bounds[1, 1], 51).float()
            xs1, xs2 = torch.meshgrid(x_lin1, x_lin2)
            x_test = torch.hstack([xs1.reshape(-1, 1), xs2.reshape(-1, 1)])
        else:
            print(bounds)
            raise ValueError("Invalid dimensions")

        self.x_test = x_test

    def _get_ts_sample_old(self, model):
        # Thompson-sample the posterior
        sampler = MaxPosteriorSampling(model=model, replacement=False)
        # TODO: rewrite this into a while loop with increasingly sparse
        # (temporatory) x_test
        try:
            with torch.no_grad():
                ts_sample = sampler(self.x_test)
        except RuntimeError:
            # print(x_train)
            print(model.covar_module.base_kernel.lengthscale.detach().numpy())
            quit()

        return ts_sample

    def _get_ts_sample(self, model):
        # Sobol sequence as opposed to a fixed grid
        sobol = SobolEngine(self.dim, scramble=True)
        x_test = sobol.draw(1000)
        # rescale
        x_test = x_test * (self.bounds[1] - self.bounds[0]) + self.bounds[0]

        sampler = "cholesky"
        with ExitStack() as es:
            if sampler == "cholesky":
                es.enter_context(gpts.max_cholesky_size(float("inf")))
            elif sampler == "ciq":
                es.enter_context(gpts.fast_computations(covar_root_decomposition=True))
                es.enter_context(gpts.max_cholesky_size(0))
                es.enter_context(gpts.ciq_samples(True))
                es.enter_context(
                    gpts.minres_tolerance(2e-3)
                )  # Controls accuracy and runtime
                es.enter_context(gpts.num_contour_quadrature(15))
            elif sampler == "lanczos":
                es.enter_context(gpts.fast_computations(covar_root_decomposition=True))
                es.enter_context(gpts.max_cholesky_size(0))
                es.enter_context(gpts.ciq_samples(False))

            # Thompson-sample the posterior
            sampler = MaxPosteriorSampling(model=model, replacement=False)
            # TODO: rewrite this into a while loop with increasingly sparse
            # (temporatory) x_test
            try:
                with torch.no_grad():
                    ts_sample = sampler(x_test)
            except RuntimeError:
                # print(x_train)
                print(model.covar_module.base_kernel.lengthscale.detach().numpy())
                print(model.covar_module.outputscale.item())
                quit()

        return ts_sample

    def run(self, model, x_train):
        ts_sample = self._get_ts_sample(model)

        # pick the second point to maximize the uncertainty of the TS sample
        # winning or losing
        def dueling_uncertainty(x):
            x = torch.atleast_2d(x)
            std_norm = Normal(torch.zeros(1), torch.ones(1))

            # output = model(torch.stack([ts_sample, x]))
            # f_post = output.mean
            # z = (f_post[0] - f_post[1]) / math.sqrt(2)

            ts_output = model(ts_sample)
            ts_mean = ts_output.mean
            x_output = model(x)
            x_mean = x_output.mean

            z = (ts_mean - x_mean) / math.sqrt(2)

            z_cdf = std_norm.cdf(z)

            return torch.abs(torch.tensor(0.5) - z_cdf)

        rival, _ = optimize_acqf(
            dueling_uncertainty,
            bounds=self.bounds,
            q=1,
            num_restarts=self.n_restarts,
            raw_samples=self.n_raw_samples,
        )

        return torch.cat([ts_sample, rival])
