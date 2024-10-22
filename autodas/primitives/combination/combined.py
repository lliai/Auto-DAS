import torch
import torch.nn.functional as F
from einops import rearrange

from ..transform import trans_batch, trans_channel


def batch_l2(f_s, f_t):
    """batch-wise l2 loss"""
    # transform to batch-wise shape
    f_s = trans_batch(f_s)
    f_t = trans_batch(f_t)

    # compute gram matrix
    G_s = torch.mm(f_s, rearrange(f_s, 'b chw -> chw  b'))
    G_t = torch.mm(f_t, rearrange(f_t, 'b chw -> chw  b'))

    # normalize
    norm_G_s = F.normalize(G_s, p=2, dim=1)
    norm_G_t = F.normalize(G_t, p=2, dim=1)

    # l2 loss
    return F.mse_loss(norm_G_s, norm_G_t)


def channel_l2(f_s, f_t):
    """inter-spatial transform for channel-wise shape"""
    # transform to channel-wise shape
    f_s = trans_channel(f_s)
    f_t = trans_channel(f_t)

    # compute gram matrix
    G_s = torch.bmm(f_s, rearrange(f_s, 'b c hw -> b hw  c'))
    G_t = torch.bmm(f_s, rearrange(f_t, 'b c hw -> b hw  c'))

    # normalize
    norm_G_t = F.normalize(G_t, p=2, dim=2)
    norm_G_s = F.normalize(G_s, p=2, dim=2)

    # l2 loss
    return F.mse_loss(norm_G_s, norm_G_t) * f_s.size(1)


def batch_kl(f_s, f_t):
    """batch-wise kl loss for logits G-L2"""
    N, C, H, W = f_s.shape
    # adopt softmax to logits
    softmax_pred_T = F.softmax(f_t.view(-1, C * W * H) / 1.0, dim=0)

    logsoftmax = torch.nn.LogSoftmax(dim=0)
    loss = torch.sum(
        softmax_pred_T * logsoftmax(f_t.view(-1, C * W * H) / 1.0) -
        softmax_pred_T * logsoftmax(f_s.view(-1, C * W * H) / 1.0)) * (1.0**2)
    return loss / (C * N)


def mask_l2(f_s, f_t):
    N, C, H, W = f_t.shape

    device = f_s.device
    mat = torch.rand((N, 1, H, W)).to(device)
    mat = torch.where(mat > 1 - 0.65, 0, 1).to(device)

    masked_fea = torch.mul(f_s, mat)

    return F.mse_loss(masked_fea, f_t) / N


def correlation(f_s, f_t):
    n, d = f_s.shape
    # normalize
    f_s_norm = (f_s - f_s.mean(0)) / f_s.std(0)
    f_t_norm = (f_t - f_t.mean(0)) / f_t.std(0)

    # compute correlation
    c_st = torch.einsum('bx,bx->x', f_s_norm, f_t_norm) / n
    c_diff = c_st - torch.ones_like(c_st)
    alpha = 1.01
    c_diff = torch.abs(c_diff)
    c_diff = c_diff.pow(2.0)
    c_diff = c_diff.pow(alpha)
    loss = torch.log2(c_diff.sum())
    return loss


def similarity(f_s, f_t):
    """ similarity loss """
    bsz = f_s.shape[0]
    bdm = f_s.shape[1]

    # inner product (normalize first and inner product)
    normft = f_t.pow(2).sum(1, keepdim=True).pow(1. / 2)
    outft = f_t.div(normft)
    normfs = f_s.pow(2).sum(1, keepdim=True).pow(1. / 2)
    outfs = f_s.div(normfs)

    # compute cosine similarity
    cos_theta = (outft * outfs).sum(1, keepdim=True)
    G_diff = 1 - cos_theta
    return (G_diff).sum() / bsz
