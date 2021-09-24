from botorch.optim import optimize_acqf
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement

from policies.base import BasePolicy


class ExpectedImprovement(BasePolicy):
    def run(self, model, x_train):
        acq_fn = qNoisyExpectedImprovement(model=model, X_baseline=x_train)
        x_next, _ = optimize_acqf(
            acq_function=acq_fn,
            bounds=self.bounds,
            q=2,
            num_restarts=self.n_restarts,
            raw_samples=self.n_raw_samples,
        )

        return x_next
