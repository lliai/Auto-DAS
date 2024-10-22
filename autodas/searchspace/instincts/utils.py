import numpy as np
import torch

from diswotv2.primitives.operations.unary_ops import to_mean_scalar


def convert_to_float(input):
    """convert to float"""
    if isinstance(input, (list, tuple)):
        if len(input) == 0:
            return -1
        return sum(convert_to_float(x) for x in input) / len(input)
    elif isinstance(input, torch.Tensor):
        return to_mean_scalar(input).item()
    elif isinstance(input, np.ndarray):
        return input.astype(float)
    elif isinstance(input, (int, float)):
        return input
    else:
        print(type(input))
        return float(input)


def convert_to_numpy(input):
    """convert to numpy array"""
    if isinstance(input, (list, tuple)):
        return np.array(
            sum([to_mean_scalar(x).detach().cpu().numpy() for x in input]))
    elif isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    elif isinstance(input, (int, float)):
        return np.array(input)
    else:
        print(type(input))
        return np.array(float(input))
