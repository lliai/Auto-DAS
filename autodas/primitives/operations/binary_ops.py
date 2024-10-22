import random
from typing import TypeVar, Union

import torch
import torch.nn.functional as F

Scalar = TypeVar('Scalar')
Vector = TypeVar('Vector')
Matrix = TypeVar('Matrix')

ALLTYPE = Union[Union[Scalar, Vector], Matrix]

# BINARY_KEYS = ('add', 'subtract', 'multiply', 'matrix_multiplication')
BINARY_KEYS = ('l1_loss', 'l2_loss', 'kl_loss', 'smooth_l1_loss')

# sample key by probability


def sample_binary_key_by_prob(probability=None):
    if probability is None:
        # the probability from large to small
        probability = [0.6, 0.3, 0.05, 0.05]
    return random.choices(
        list(range(len(BINARY_KEYS))), weights=probability, k=1)[0]


# binary operator


def add(A: ALLTYPE, B: ALLTYPE) -> ALLTYPE:
    """add = (A + B)"""
    return A + B


def mean(A: ALLTYPE, B: ALLTYPE) -> ALLTYPE:
    """mean = (A + B) / 2"""
    return (A + B) / 2


# SCALAR_DIFF_OP


def subtract(A: ALLTYPE, B: ALLTYPE) -> ALLTYPE:
    """subtract = (A - B)"""
    return A - B


def multiply(A: ALLTYPE, B: ALLTYPE) -> ALLTYPE:
    """multiply = (A * B)"""
    return A * B


def matrix_multiplication(A: Matrix, B: Matrix):
    """matrix_multiplication = A @ B"""
    return A @ B


def lesser_than(A: ALLTYPE, B: ALLTYPE) -> bool:
    """lesser_than = (A < B)"""
    return (A < B).float()


def greater_than(A: ALLTYPE, B: ALLTYPE) -> bool:
    """greater_than = (A > B)"""
    return (A > B).float()


def equal_to(A: ALLTYPE, B: ALLTYPE) -> bool:
    """equal_to = (A == B)"""
    return (A == B).float()


def hamming_distance(A: ALLTYPE, B: ALLTYPE) -> Scalar:
    """hamming_distance = (A != B)"""
    value = torch.tensor([0], dtype=A.dtype)
    A = torch.heaviside(A, values=value)
    B = torch.heaviside(B, values=value)
    return add(A != B)


def pairwise_distance(A: Matrix, B: Matrix) -> Vector:
    """pairwise_distance = (A - B) ** 2"""
    return F.pairwise_distance(A, B, p=2)


def kl_divergence(A: ALLTYPE, B: ALLTYPE) -> Scalar:
    """kl_divergence = A * log(A / B)"""
    return torch.nn.KLDivLoss('batchmean')(A, B)


def cosine_similarity(A: Matrix, B: Matrix) -> Scalar:
    """cosine_similarity = A @ B / (|A| * |B|)"""
    A = A.reshape(A.shape[0], -1)
    B = A.reshape(B.shape[0], -1)
    C = torch.nn.CosineSimilarity()(A, B)
    return torch.add(C)


def mse_loss(A: ALLTYPE, B: ALLTYPE) -> Scalar:
    """mse_loss = l2_loss = (A - B) ** 2"""
    return F.mse_loss(A, B)


def l1_loss(A: ALLTYPE, B: ALLTYPE) -> Scalar:
    """l1_loss = (A - B).abs()"""
    return F.l1_loss(A, B)


def l2_loss(A: ALLTYPE, B: ALLTYPE) -> Scalar:
    """mse_loss = l2_loss = (A - B) ** 2"""
    return F.mse_loss(A, B)


def kl_loss(A: ALLTYPE, B: ALLTYPE) -> Scalar:
    """kl_loss = kl_divergence = A * log(A / B)"""
    return F.kl_div(A, B, reduction='batchmean')


def smooth_l1_loss(A: ALLTYPE, B: ALLTYPE) -> Scalar:
    """smooth_l1_loss = (A - B).abs()"""
    return F.smooth_l1_loss(A, B)


def binary_operation(A, B, idx: Union[str, int] = None):
    # 10
    if idx is None:
        idx = random.choice(range(len(BINARY_KEYS)))
    elif isinstance(idx, str):
        idx = BINARY_KEYS.index(idx)
    elif isinstance(idx, int):
        pass
    else:
        raise ValueError('idx must be str or int')

    assert idx < len(BINARY_KEYS)

    binaries = {
        'add': add,
        'subtract': subtract,
        'multiply': multiply,
        'matrix_multiplication': matrix_multiplication,
        'lesser_than': lesser_than,
        'greater_than': greater_than,
        'equal_to': equal_to,
        'hamming_distance': hamming_distance,
        'kl_divergence': kl_divergence,
        'cosine_similarity': cosine_similarity,
        'pairwise_distance': pairwise_distance,
        'l1_loss': l1_loss,
        'l2_loss': l2_loss,
        'mse_loss': mse_loss,
        'kl_loss': kl_loss,
        'smooth_l1_loss': smooth_l1_loss,
    }
    return binaries[BINARY_KEYS[idx]](A, B)
