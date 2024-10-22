import random
from typing import TypeVar, Union

import torch
import torch.nn.functional as F

Scalar = TypeVar('Scalar')
Vector = TypeVar('Vector')
Matrix = TypeVar('Matrix')

ALLTYPE = Union[Union[Scalar, Vector], Matrix]

UNARY_KEYS = ('log', 'abslog', 'abs', 'exp', 'normalize', 'tanh', 'square',
              'relu', 'invert', 'frobenius_norm', 'normalized_sum', 'l1_norm',
              'softmax', 'sigmoid', 'logsoftmax', 'sqrt', 'revert', 'no_op')
AGGRERATE_KEYS = ('local_max_pooling', 'local_min_pooling',
                  'channel_wise_mean', 'spatial_wise_mean', 'no_op')


# sample key by probability
def sample_unary_key_by_prob(probability=None):
    if probability is None:
        # other than the last one, the rest are the same small
        probability = [0.1] * (len(UNARY_KEYS) - 1) + [0.2]
    return random.choices(
        list(range(len(UNARY_KEYS))), weights=probability, k=1)[0]


# unary operation
def no_op(A: ALLTYPE) -> ALLTYPE:
    """no_op = A"""
    return A


def log(A: ALLTYPE) -> ALLTYPE:
    """log = log(A)"""
    # A[A <= 0] == 1
    # return torch.log(A + 1e-9)
    return torch.sign(A) * torch.log(torch.abs(A) + 1e-9)


def square(A: ALLTYPE) -> ALLTYPE:
    """square = A^2"""
    return torch.pow(A, 2)


def revert(A: ALLTYPE) -> ALLTYPE:
    """revert = -A"""
    return A * -1


def min_max_normalize(A: ALLTYPE) -> ALLTYPE:
    """min_max_normalize = (A - A_min) / (A_max - A_min)"""
    A_min, A_max = A.min(), A.max()
    return (A - A_min) / (A_max - A_min + 1e-9)


def abslog(A: ALLTYPE) -> ALLTYPE:
    """abslog = log(|A|)"""
    A[A == 0] = 1
    A = torch.abs(A)
    return torch.log(A)


def abs(A: ALLTYPE) -> ALLTYPE:
    """abs = |A|"""
    return torch.abs(A)


def sqrt(A: ALLTYPE) -> ALLTYPE:
    """sqrt = sqrt(|A|)"""
    A[A <= 0] = 0
    return torch.sqrt(A)


def exp(A: ALLTYPE) -> ALLTYPE:
    """exp = exp(A)"""
    return torch.exp(A)


def normalize(A: ALLTYPE) -> ALLTYPE:
    """normalize = (A - mean(A)) / std(A)"""
    m = torch.mean(A)
    s = torch.std(A)
    C = (A - m) / s
    C[C != C] = 0
    return C


def relu(A: ALLTYPE) -> ALLTYPE:
    """relu = max(0, A)"""
    return F.relu(A)


def tanh(A: ALLTYPE) -> ALLTYPE:
    """tanh = tanh(A)"""
    return torch.tanh(A)


def sign(A: ALLTYPE) -> ALLTYPE:
    """sign = sign(A)"""
    return F.softsign(A)


def invert(A: ALLTYPE) -> ALLTYPE:
    """invert = 1 / (A + 1e-9)"""
    if isinstance(A, (int, float)) and A == 0:
        raise ZeroDivisionError
    return 1 / (A + 1e-9)


def frobenius_norm(A: ALLTYPE) -> Scalar:
    """frobenius_norm = ||A||_F"""
    return torch.norm(A, p='fro')


def normalized_sum(A: ALLTYPE) -> Scalar:
    """normalized_sum = sum(A) / (numel(A) + 1e-9)"""
    return torch.sum(A) / (A.numel() + 1e-9)


def l1_norm(A: ALLTYPE) -> Scalar:
    """l1_norm = sum(|A|) / (numel(A) + 1e-9)"""
    return torch.sum(torch.abs(A)) / (A.numel() + 1e-9)


def p_dist(A: Matrix) -> Vector:
    """p_dist = pdist(A)"""
    return F.pdist(A)


def softmax(A: ALLTYPE) -> ALLTYPE:
    """softmax = softmax(A)"""
    return F.softmax(A, dim=0)


def logsoftmax(A: ALLTYPE) -> ALLTYPE:
    """logsoftmax = logsoftmax(A)"""
    return F.log_softmax(A, dim=0)


def sigmoid(A: ALLTYPE) -> ALLTYPE:
    """sigmoid = sigmoid(A)"""
    return torch.sigmoid(A)


def slogdet(A: Matrix) -> Scalar:
    """slogdet = slogdet(A)"""
    sign, value = torch.linalg.slogdet(A)
    return value


def to_mean_scalar(A: ALLTYPE) -> Scalar:
    """to_mean_scalar = mean(A)"""
    return torch.mean(A)


def to_sum_scalar(A: ALLTYPE) -> Scalar:
    """to_sum_scalar = sum(A)"""
    return torch.sum(A)


def to_std_scalar(A: ALLTYPE) -> Scalar:
    """to_std_scalar = std(A)"""
    return torch.std(A)


def to_var_scalar(A: ALLTYPE) -> Scalar:
    """to_var_scalar = var(A)"""
    return torch.var(A)


def to_min_scalar(A: ALLTYPE) -> Scalar:
    """to_min_scalar = min(A)"""
    return torch.min(A)


def to_max_scalar(A: ALLTYPE) -> Scalar:
    """to_max_scalar = max(A)"""
    return torch.max(A)


def to_sqrt_scalar(A: ALLTYPE) -> Scalar:
    """to_sqrt_scalar = sqrt(A)"""
    A[A <= 0] = 0
    return torch.sqrt(A)


def channel_wise_mean(A: Matrix) -> Vector:
    """channel_wise_mean = mean_{c, nhw}"""
    # mean_{c}
    return torch.mean(A, dim=1)


def spatial_wise_mean(A: Matrix) -> Vector:
    """spatial_wise_mean = mean_{nhw}"""
    # mean_{nhw}
    return torch.mean(A, dim=(0, 2, 3))


def local_max_pooling(A: Matrix) -> Matrix:
    """local_max_pooling = max_{3x3}"""
    return F.max_pool2d(A, kernel_size=3, stride=1, padding=1)


def local_min_pooling(A: Matrix) -> Matrix:
    """local_min_pooling = min_{3x3}"""
    return F.max_pool2d(-A, kernel_size=3, stride=1, padding=1) * -1


def gram_matrix(A: Matrix) -> Matrix:
    """https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
    """
    assert len(A.shape) == 4, 'Input shape is invalid.'
    a, b, c, d = A.size()
    feature = A.view(a * b, c * d)
    G = torch.mm(feature, feature.t())
    return G.div(a * b * c * d)


def unary_operation(A, idx: Union[str, int] = None):
    if idx is None:
        idx = random.choice(range(len(UNARY_KEYS)))
    elif isinstance(idx, str):
        idx = UNARY_KEYS.index(idx)
    elif isinstance(idx, int):
        pass  # do nothing
    else:
        raise TypeError(f'idx must be str or int, but got {type(idx)})')

    assert idx < len(UNARY_KEYS)

    unaries = {
        'log': log,
        'abslog': abslog,
        'abs': abs,
        'pow': pow,
        'exp': exp,
        'normalize': normalize,
        'relu': relu,
        'sign': sign,
        'invert': invert,
        'frobenius_norm': frobenius_norm,
        'normalized_sum': normalized_sum,
        'l1_norm': l1_norm,
        'softmax': softmax,
        'sigmoid': sigmoid,
        'p_dist': p_dist,
        'to_mean_scalar': to_mean_scalar,
        'to_sum_scalar': to_sum_scalar,
        'to_std_scalar': to_std_scalar,
        'to_min_scalar': to_min_scalar,
        'to_max_scalar': to_max_scalar,
        'to_sqrt_scalar': to_sqrt_scalar,
        'gram_matrix': gram_matrix,
        'logsoftmax': logsoftmax,
        'sqrt': sqrt,
        'min_max_normalize': min_max_normalize,
        'revert': revert,
        'no_op': no_op,
        'tanh': tanh,
    }
    return unaries[UNARY_KEYS[idx]](A)


def aggeregation_operation(A, idx: Union[str, int] = None):
    if idx is None:
        idx = random.choice(range(len(AGGRERATE_KEYS)))
    elif isinstance(idx, str):
        idx = AGGRERATE_KEYS.index(idx)
    elif isinstance(idx, int):
        pass  # do nothing
    else:
        raise TypeError('idx must be str or int.')

    assert idx < len(AGGRERATE_KEYS)

    aggregations = {
        'channel_wise_mean': channel_wise_mean,
        'spatial_wise_mean': spatial_wise_mean,
        'local_max_pooling': local_max_pooling,
        'local_min_pooling': local_min_pooling,
        'no_op': no_op,
    }
    return aggregations[AGGRERATE_KEYS[idx]](A)
