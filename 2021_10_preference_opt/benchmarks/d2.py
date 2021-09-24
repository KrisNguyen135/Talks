import torch
from math import pi


bounds1 = torch.stack([torch.tensor([-2, -2]), torch.tensor([2, 2])]).float()
x_opt1 = torch.tensor([0, 0]).float()


def three_hump_camel(x):
    x = torch.atleast_2d(x)

    return (
        -(
            2 * x[:, 0] ** 2
            - 1.05 * x[:, 0] ** 4
            + x[:, 0] ** 6 / 6
            + x[:, 0] * x[:, 1]
            + x[:, 1] ** 2
        )
        / 500
        - 2
    )


bounds2 = torch.stack([torch.tensor([-10, -10]), torch.tensor([10, 10])]).float()
x_opt2 = torch.tensor([0, 0]).float()


def ackley(x):
    x = torch.atleast_2d(x)

    return -(
        -20 * torch.exp(-0.2 * torch.sqrt((x[:, 0] ** 2 + x[:, 1] ** 2) / 2))
        - torch.exp(
            (
                torch.cos(2 * pi * torch.cos(x[:, 0]))
                + torch.cos(2 * pi * torch.cos(x[:, 1]))
            )
            / 2
        )
        + 12.5
    )


bounds3 = torch.stack([torch.tensor([-5, -5]), torch.tensor([5, 5])]).float()
x_opt3 = torch.tensor([0, 0]).float()


def griewank(x):
    x = torch.atleast_2d(x)

    # return (x[:, 0] ** 2 + x[:, 1] ** 2) / 4000 \
    #     - torch.cos(x[:, 0]) * torch.cos(x[:, 1] / torch.sqrt(torch.tensor(2)))

    return (
        torch.cos(x[:, 0]) * torch.cos(x[:, 1] / torch.sqrt(torch.tensor(2)))
        - (x[:, 0] ** 2 + x[:, 1] ** 2) / 4000
    )


def get_benchmark_2d(name):
    if name == "three-hump camel":
        return three_hump_camel, bounds1, x_opt1
    elif name == "ackley":
        return ackley, bounds2, x_opt2
    elif name == "griewank":
        return griewank, bounds3, x_opt3
    else:
        raise ValueError("Unknown function name.")
