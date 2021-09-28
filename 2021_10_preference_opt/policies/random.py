from policies.base import BasePolicy
import torch


class RandomPolicy(BasePolicy):
    def run(self):
        return (torch.rand(2, self.bounds.shape[0]) + self.bounds[0]) * (
            self.bounds[1] - self.bounds[0]
        )
