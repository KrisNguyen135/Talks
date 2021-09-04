import torch
from math import pi


bounds1 = torch.stack(
    [torch.tensor([-2.048, -2.048, -2.048]), torch.tensor([2.048, 2.048, 2.048])]
).float()
x_opt1 = torch.tensor([1, 1, 1]).float()

def rosenbrock(x):
    x = torch.atleast_2d(x)

    return - (
        100 * (x[:, 1] - x[:, 0] ** 2) ** 2 + (x[:, 0] - 1) ** 2
        + 100 * (x[:, 2] - x[:, 1] ** 2) ** 2 + (x[:, 1] - 1) ** 2
    )


bounds2 = torch.stack(
    [torch.tensor([-10, -10, -10]), torch.tensor([10, 10, 10])]
).float()
x_opt2 = torch.tensor([0, 0, 0]).float()

def ackley(x):
    x = torch.atleast_2d(x)

    return - (
        -20 * torch.exp(
            -0.2 * torch.sqrt((x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2) / 2)
        )
        - torch.exp(
            (
                torch.cos(2 * pi * torch.cos(x[:, 0]))
                + torch.cos(2 * pi * torch.cos(x[:, 1]))
                + torch.cos(2 * pi * torch.cos(x[:, 2]))
            ) / 2
        )
        + 12.5
    )


bounds3 = torch.stack([torch.tensor([-5, -5, -5]), torch.tensor([5, 5, 5])]).float()
x_opt3 = torch.tensor([0, 0, 0]).float()

def griewank(x):
    x = torch.atleast_2d(x)

    return (
        torch.cos(x[:, 0])
        * torch.cos(x[:, 1] / torch.sqrt(torch.tensor(2)))
        * torch.cos(x[:, 2] / torch.sqrt(torch.tensor(3)))
    ) - (x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2) / 4000


def get_benchmark_3d(name):
    if name == 'rosenbrock':
        return rosenbrock, bounds1, x_opt1
    elif name == 'ackley':
        return ackley, bounds2, x_opt2
    elif name == 'griewank':
        return griewank, bounds3, x_opt3
    else:
        raise ValueError('Unknown function name.')
