import torch
import torch.nn.functional as F
from torch import Tensor

from .registry import register_distance


@register_distance
def mse_loss(f_s: Tensor, f_t: Tensor) -> Tensor:
    """mse_loss = l2_loss = (f_s - f_t) ** 2"""
    return F.mse_loss(f_s, f_t)


@register_distance
def l1_loss(f_s: Tensor, f_t: Tensor) -> Tensor:
    """l1_loss = (f_s - f_t).abs()"""
    return F.l1_loss(f_s, f_t)


@register_distance
def l2_loss(f_s: Tensor, f_t: Tensor) -> Tensor:
    """mse_loss = l2_loss = (f_s - f_t) ** 2"""
    return F.mse_loss(f_s, f_t)


@register_distance
def kl_loss(f_s: Tensor, f_t: Tensor) -> Tensor:
    """kl_loss = kl_divergence = f_s * log(f_s / f_t)"""
    return F.kl_div(f_s, f_t, reduction='batchmean')


@register_distance
def kl_T1(f_s: Tensor, f_t: Tensor, T: float = 1) -> Tensor:
    """kl_loss = kl_divergence = f_s * log(f_s / f_t)"""
    return F.kl_div(
        F.log_softmax(f_s / T, dim=1),
        F.softmax(f_t / T, dim=1),
        reduction='batchmean') * T * T


@register_distance
def kl_T4(f_s: Tensor, f_t: Tensor, T: float = 4) -> Tensor:
    """kl_loss = kl_divergence = f_s * log(f_s / f_t)"""
    return F.kl_div(
        F.log_softmax(f_s / T, dim=1),
        F.softmax(f_t / T, dim=1),
        reduction='batchmean') * T * T


@register_distance
def kl_T8(f_s: Tensor, f_t: Tensor, T: float = 8) -> Tensor:
    """kl_loss = kl_divergence = f_s * log(f_s / f_t)"""
    return F.kl_div(
        F.log_softmax(f_s / T, dim=1),
        F.softmax(f_t / T, dim=1),
        reduction='batchmean') * T * T


@register_distance
def smooth_l1_loss(f_s: Tensor, f_t: Tensor) -> Tensor:
    """smooth_l1_loss = (f_s - f_t).abs()"""
    return F.smooth_l1_loss(f_s, f_t)


@register_distance
def cosine_similarity(f_s, f_t, eps=1e-8):
    """cosine_similarity = f_s * f_t / (|f_s| * |f_t|)"""
    return F.cosine_similarity(f_s, f_t, eps=eps).mean()


@register_distance
def pearson_correlation(f_s, f_t, eps=1e-8):
    """pearson_correlation = (f_s - mean(f_s)) * (f_t - mean(f_t)) / (|f_s - mean(f_s)| * |f_t - mean(f_t)|)"""

    def cosine(f_s, f_t, eps=1e-8):
        return (f_s * f_t).sum(1) / (f_s.norm(dim=1) * f_t.norm(dim=1) + eps)

    return 1 - cosine(f_s - f_s.mean(1).unsqueeze(1),
                      f_t - f_t.mean(1).unsqueeze(1), eps).mean()


@register_distance
def pairwise_distance(f_s: Tensor, f_t: Tensor):
    """pairwise_distance = (f_s - f_t) ** 2"""
    return F.pairwise_distance(f_s, f_t, p=2).mean()


@register_distance
def subtract(f_s: Tensor, f_t: Tensor) -> Tensor:
    """subtract = f_s - f_t"""
    return (f_s - f_t).mean()


@register_distance
def multiply(f_s: Tensor, f_t: Tensor) -> Tensor:
    """multiply = f_s * f_t"""
    return (f_s * f_t).mean()


@register_distance
def matrix_multiplication(f_s: Tensor, f_t: Tensor):
    """Tensor_multiplication = f_s @ f_t"""
    if len(f_s.shape) == 2:
        # logits （A,f_t): [N, C]
        return torch.einsum('nc,nc->', f_s, f_t)
    elif len(f_s.shape) == 4:
        # for features （A,f_t): [N, C, H, W]
        return torch.einsum('ncwh,nchw->', f_s, f_t.transpose(2, 3))
    elif len(f_s.shape) == 3:
        # for middle features （A,f_t): [N, C, M]
        return torch.einsum('ncm,nmc->', f_s, f_t.transpose(1, 2))
    else:
        raise f'Not support {f_s.shape}'


@register_distance
def lesser_than(f_s: Tensor, f_t: Tensor):
    """lesser_than = f_s < f_t"""
    return (f_s < f_t).float().mean()
