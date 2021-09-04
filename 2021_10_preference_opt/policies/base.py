class BasePolicy:
    def __init__(self, bounds, n_restarts=10, n_raw_samples=20):
        self.bounds = bounds
        self.n_restarts = n_restarts
        self.n_raw_samples = n_raw_samples

    def run(self):
        raise NotImplementedError
