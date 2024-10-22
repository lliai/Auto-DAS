from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from .registry import register_transform


@register_transform
def trans_multi_scale_r1(f):
    """transform with multi-scale distillation with reduce ratio of 1"""
    if len(f.shape) != 4:
        return f

    return reduce(f, 'b c (h1 h2) (w1 w2) -> b c h1 w1', 'max', h2=1, w2=1)


@register_transform
def trans_multi_scale_r2(f):
    """transform with multi-scale distillation with reduce ratio of 2"""
    if len(f.shape) != 4:
        return f
    return reduce(f, 'b c (h1 h2) (w1 w2) -> b c h1 w1', 'max', h2=2, w2=2)


@register_transform
def trans_multi_scale_r4(f):
    """transform with multi-scale distillation with reduce ratio of 4"""
    if len(f.shape) != 4:
        return f
    return reduce(f, 'b c (h1 h2) (w1 w2) -> b c h1 w1', 'max', h2=4, w2=4)


@register_transform
def trans_local_s1(f):
    """transform with local features distillation with spatial size of 1"""
    if len(f.shape) != 4:
        return f
    f = rearrange(f, 'b c (h hp) (w wp) -> b (c h w) hp wp', hp=1, wp=1)
    return f.squeeze(-1).squeeze(-1)


@register_transform
def trans_local_s2(f):
    """transform with local features distillation with spatial size of 1"""
    if len(f.shape) != 4:
        return f

    return rearrange(f, 'b c (h hp) (w wp) -> b (c h w) hp wp', hp=2, wp=2)


@register_transform
def trans_local_s4(f):
    """transform with local features distillation with spatial size of 1"""
    if len(f.shape) != 4:
        return f

    return rearrange(f, 'b c (h hp) (w wp) -> b (c h w) hp wp', hp=4, wp=4)


@register_transform
def trans_batch(f):
    """transform with batch-wise shape"""
    if len(f.shape) == 2:
        return f
    elif len(f.shape) == 3:
        return rearrange(f, 'b c h -> b (c h)')
    elif len(f.shape) == 4:
        return rearrange(f, 'b c h w -> b (c h w)')


@register_transform
def trans_channel(f):
    """transform with channel-wise shape"""
    if len(f.shape) in {2, 3}:
        return f
    elif len(f.shape) == 4:
        return rearrange(f, 'b c h w -> b c (h w)')


@register_transform
def trans_mask(f, threshold=0.65):
    """transform with mask"""
    if len(f.shape) in {2, 3}:
        # logits
        return f
    N, C, H, W = f.shape
    device = f.device
    mat = torch.rand((N, 1, H, W)).to(device)
    mat = torch.where(mat > 1 - threshold, 0, 1).to(device)
    return torch.mul(f, mat)


@register_transform
def trans_satt(f, T=0.5):
    """transform with spatial attention"""
    if len(f.shape) in {2, 3}:
        # logits
        return f

    N, C, H, W = f.shape
    value = torch.abs(f)
    fea_map = value.mean(axis=1, keepdim=True)
    # Bs*W*H
    S_attention = (H * W * F.softmax(
        (fea_map / T).view(N, -1), dim=1)).view(N, H, W)
    return S_attention.unsqueeze(dim=-1)


@register_transform
def trans_natt(f, T=0.5):
    """transform from the N dim"""
    if len(f.shape) == 2:
        N, C = f.shape
    elif len(f.shape) == 4:
        N, C, H, W = f.shape
    elif len(f.shape) == 3:
        N, C, M = f.shape

    # apply softmax to N dim
    return N * F.softmax(f / T, dim=0)


@register_transform
def trans_catt(f, T=0.5):
    """transform with channel attention"""
    if len(f.shape) == 2:
        # logits
        N, C = f.shape
        # apply softmax to C dim
        return C * F.softmax(f / T, dim=1)
    elif len(f.shape) == 3:
        N, C, M = f.shape
        return C * F.softmax(f / T, dim=1)
    elif len(f.shape) == 4:
        N, C, H, W = f.shape
        value = torch.abs(f)
        # Bs*C
        channel_map = value.mean(
            axis=2, keepdim=False).mean(
                axis=2, keepdim=False)
        C_attention = C * F.softmax(channel_map / T, dim=1)
        return C_attention.unsqueeze(dim=-1).unsqueeze(dim=-1)
    else:
        raise f'invalid shape {f.shape}'


@register_transform
def trans_drop(f, p=0.1):
    """transform with dropout"""
    return F.dropout2d(f, p)


@register_transform
def trans_nop(f):
    """no operation transform """
    return f


@register_transform
def trans_bmm(f):
    """transform with gram matrix -> b, c, c"""
    if len(f.shape) == 2:
        return f
    elif len(f.shape) == 4:
        return torch.bmm(
            rearrange(f, 'b c h w -> b c (h w)'),
            rearrange(f, 'b c h w -> b (h w) c'))
    elif len(f.shape) == 3:
        return torch.bmm(
            rearrange(f, 'b c m -> b c m'), rearrange(f, 'b c m -> b m c'))
    else:
        raise f'invalide shape {f.shape}'


@register_transform
def trans_mm(f):
    """transform with gram matrix -> b, b"""
    if len(f.shape) == 2:
        return f
    elif len(f.shape) == 3:
        return torch.mm(
            rearrange(f, 'b c m -> b (c m)'), rearrange(f, 'b c m -> (c m) b'))
    elif len(f.shape) == 4:
        return torch.mm(
            rearrange(f, 'b c h w -> b (c h w)'),
            rearrange(f, 'b c h w -> (c h w) b'))
    else:
        raise f'invalide shape {f.shape}'


@register_transform
def trans_norm_HW(f):
    """transform with l2 norm in HW dim"""
    if len(f.shape) == 2:
        return f
    elif len(f.shape) == 3:
        return F.normalize(f, p=2, dim=2)
    elif len(f.shape) == 4:
        return F.normalize(f, p=2, dim=(2, 3))
    else:
        raise f'invalide shape {f.shape}'


@register_transform
def trans_norm_C(f):
    """transform with l2 norm in C dim"""
    return F.normalize(f, p=2, dim=1)


@register_transform
def trans_norm_N(f):
    """ transform with l2 norm in N dim"""
    return F.normalize(f, p=2, dim=0)


@register_transform
def trans_softmax_N(f):
    """transform with softmax in 0 dim"""
    return F.softmax(f, dim=0)


@register_transform
def trans_softmax_C(f):
    """transform with softmax in 1 dim"""
    return F.softmax(f, dim=1)


@register_transform
def trans_softmax_HW(f):
    """transform with softmax in 2,3 dim"""
    if len(f.shape) == 2:
        return f

    if len(f.shape) == 4:
        N, C, H, W = f.shape
        f = f.reshape(N, C, -1)

    assert len(f.shape) == 3
    return F.softmax(f, dim=2)


@register_transform
def trans_logsoftmax_N(f):
    """transform with logsoftmax"""
    return F.log_softmax(f, dim=1)


@register_transform
def trans_logsoftmax_C(f):
    """transform with logsoftmax"""
    return F.log_softmax(f, dim=1)


@register_transform
def trans_logsoftmax_HW(f):
    """transform with logsoftmax"""
    if len(f.shape) == 2:
        return f

    if len(f.shape) == 4:
        N, C, H, W = f.shape
        f = f.reshape(N, C, -1)

    assert len(f.shape) == 3
    return F.log_softmax(f, dim=2)


@register_transform
def trans_sqrt(f):
    """transform with sqrt"""
    return torch.sqrt(f)


@register_transform
def trans_log(f):
    """transform with log"""
    return torch.sign(f) * torch.log(torch.abs(f) + 1e-9)


@register_transform
def trans_pow2(f):
    """transform with ^2"""
    return torch.pow(f, 2)


@register_transform
def trans_pow4(f):
    """transform with ^4"""
    return torch.pow(f, 4)


@register_transform
def trans_min_max_normalize(f):
    """transform with min-max normalize"""
    A_min, A_max = f.min(), f.max()
    return (f - A_min) / (A_max - A_min + 1e-9)


@register_transform
def trans_abs(f):
    """transform with abs"""
    return torch.abs(f)


@register_transform
def trans_sigmoid(f):
    """transform with sigmoid"""
    return torch.sigmoid(f)


@register_transform
def trans_swish(f):
    """transform with swish"""
    return f * torch.sigmoid(f)


@register_transform
def trans_tanh(f):
    """transform with tanh"""
    return torch.tanh(f)


@register_transform
def trans_relu(f):
    """transform with relu"""
    return F.relu(f)


@register_transform
def trans_leaky_relu(f):
    """transform with leaky relu"""
    return F.leaky_relu(f)


@register_transform
def trans_mish(f):
    """transform with mish"""
    return f * torch.tanh(F.softplus(f))


@register_transform
def trans_exp(f):
    """transform with exp"""
    return torch.exp(f)


@register_transform
def trans_scale(f):
    """transform 0-1"""
    return (f + 1.0) / 2.0


@register_transform
def trans_batchnorm(f):
    """transform with batchnorm"""
    if len(f.shape) in {2, 3}:
        bn = nn.BatchNorm1d(f.shape[1]).to(f.device)
    elif len(f.shape) == 4:
        bn = nn.BatchNorm2d(f.shape[1]).to(f.device)
    return bn(f)
