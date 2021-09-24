from policies.base import BasePolicy

import torch

from botorch.optim import optimize_acqf
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement


class IncumbentPolicy(BasePolicy):
    def find_incumbent_old(self, model):
        post_mean = lambda x: model(x).mean

        incumbent, incumbent_val = optimize_acqf(
            post_mean,
            bounds=self.bounds,
            q=1,
            num_restarts=self.n_restarts,
            raw_samples=self.n_raw_samples,
        )

        return incumbent, incumbent_val

    def find_incumbent(self, model, x_train):
        with torch.no_grad():
            mean_train = model(x_train).mean

        incumbent_ind = torch.argmax(mean_train)

        return x_train[incumbent_ind].unsqueeze(0), mean_train[incumbent_ind].unsqueeze(
            0
        )

    def find_rival(self, model, x_train):
        raise NotImplementedError

    def run(self, model, x_train):
        incumbent, incumbent_val = self.find_incumbent(model, x_train)
        rival = self.find_rival(model, x_train)

        return torch.cat([incumbent, rival])


class IncumbentExpectedImprovement(IncumbentPolicy):
    def find_rival(self, model, x_train):
        acq_fn = qNoisyExpectedImprovement(model=model, X_baseline=x_train)
        rival, _ = optimize_acqf(
            acq_function=acq_fn,
            bounds=self.bounds,
            q=1,
            num_restarts=self.n_restarts,
            raw_samples=self.n_raw_samples,
        )

        return rival
