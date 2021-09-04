import torch


bounds1 = torch.stack([torch.tensor([-2.5]), torch.tensor([7.5])]).float()
x_opt1 = torch.tensor([5.145735]).float()

def f1(x):
    return - torch.sin(x) - torch.sin(10 / 3 * x)


bounds2 = torch.stack([torch.tensor([0]), torch.tensor([1.2])]).float()
x_opt2 = torch.tensor([0.96609]).float()

def f2(x):
    return (1.4 - 3 * x) * torch.sin(18 * x)


bounds3 = torch.stack([torch.tensor([0]), torch.tensor([10])]).float()
x_opt3 = torch.tensor([7.9787]).float()

def f3(x):
    return x * torch.sin(x)


def get_benchmark_1d(name):
    if name == 'f1':
        return f1, bounds1, x_opt1
    elif name == 'f2':
        return f2, bounds2, x_opt2
    elif name == 'f3':
        return f3, bounds3, x_opt3
    else:
        raise ValueError('Unknown function name.')
