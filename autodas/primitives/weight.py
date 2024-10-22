import torch.nn.functional as F
from torch import Tensor, tensor

from .registry import register_weight

# f represent the output from distance function
# t represent the output from transform function from teacher branch


@register_weight
def w1_teacher(f: Tensor, t: Tensor = None, s: Tensor = None):
    return f


# def w1_teacher(f: Tensor, t: Tensor = None, s: Tensor = None):
#     """w1_teacher = f * 1 * t"""
#     if t is None and s is None:
#         return f * tensor(1).to(f.device)

#     if len(t.shape) == 1:
#         N = t.shape
#     elif len(t.shape) == 2:
#         N, C = t.shape
#     elif len(t.shape) == 3:
#         N, C, M = t.shape
#     elif len(t.shape) == 4:
#         N, C, H, W = t.shape
#     else:
#         raise f'invalid shape {t.shape}'

#     t = F.softmax(t.view(N, -1) / 1.0, dim=0).mean()
#     return f * tensor(1).to(f.device) * t

# @register_weight
# def w5_teacher(f: Tensor, t: Tensor = None, s: Tensor = None):
#     """w5_teacher = f * 5 * t"""
#     if t is None and s is None:
#         return f * tensor(5).to(f.device)
#     if len(t.shape) == 1:
#         N = t.shape
#     elif len(t.shape) == 2:
#         N, C = t.shape
#     elif len(t.shape) == 3:
#         N, C, M = t.shape
#     elif len(t.shape) == 4:
#         N, C, H, W = t.shape
#     else:
#         raise f'invalid shape {t.shape}'

#     t = F.softmax(t.view(N, -1) / 1.0, dim=0).mean()
#     return f * tensor(5).to(f.device) * t

# @register_weight
# def w25_teacher(f: Tensor, t: Tensor = None, s: Tensor = None):
#     """w25_teacher = f * 25 * t"""
#     if t is None and s is None:
#         return f * tensor(25).to(f.device)
#     if len(t.shape) == 1:
#         N = t.shape
#     elif len(t.shape) == 2:
#         N, C = t.shape
#     elif len(t.shape) == 3:
#         N, C, M = t.shape
#     elif len(t.shape) == 4:
#         N, C, H, W = t.shape
#     else:
#         raise f'invalid shape {t.shape}'

#     t = F.softmax(t.view(N, -1) / 1.0, dim=0).mean()
#     return f * tensor(25).to(f.device) * t

# @register_weight
# def w50_teacher(f: Tensor, t: Tensor = None, s: Tensor = None):
#     """w50_teacher = f * 50 * t"""
#     if t is None and s is None:
#         return f * tensor(50).to(f.device)
#     if len(t.shape) == 1:
#         N = t.shape
#     elif len(t.shape) == 2:
#         N, C = t.shape
#     elif len(t.shape) == 3:
#         N, C, M = t.shape
#     elif len(t.shape) == 4:
#         N, C, H, W = t.shape
#     else:
#         raise f'invalid shape {t.shape}'

#     t = F.softmax(t.view(N, -1) / 1.0, dim=0).mean()
#     return f * tensor(50).to(f.device) * t

# @register_weight
# def w100_teacher(f: Tensor, t: Tensor = None, s: Tensor = None):
#     """w100_teacher = f * 100 * t"""
#     if t is None and s is None:
#         return f * tensor(100).to(f.device)

#     if len(t.shape) == 1:
#         N = t.shape
#     elif len(t.shape) == 2:
#         N, C = t.shape
#     elif len(t.shape) == 3:
#         N, C, M = t.shape
#     elif len(t.shape) == 4:
#         N, C, H, W = t.shape
#     else:
#         raise f'invalid shape {t.shape}'

#     t = F.softmax(t.view(N, -1) / 1.0, dim=0).mean()
#     return f * tensor(100).to(f.device) * t

# @register_weight
# def w1_teacher_student(f: Tensor, t: Tensor = None, s: Tensor = None):
#     """w1_teacher_student = f * 1 * t * s"""
#     if t is None and s is None:
#         return f * tensor(1).to(f.device)

#     if len(t.shape) == 1:
#         N = t.shape
#     elif len(t.shape) == 2:
#         N, C = t.shape
#     elif len(t.shape) == 3:
#         N, C, M = t.shape
#     elif len(t.shape) == 4:
#         N, C, H, W = t.shape
#     else:
#         raise f'invalid shape {t.shape}'

#     # compute cosine similarity for t and s
#     t = F.cosine_similarity(t.view(N, -1), s.view(N, -1), dim=0).mean()

#     return f * tensor(1).to(f.device) * t

# @register_weight
# def w5_teacher_student(f: Tensor, t: Tensor = None, s: Tensor = None):
#     """w5_teacher_student = f * 5 * t * s"""
#     if t is None and s is None:
#         return f * tensor(5).to(f.device)
#     if len(t.shape) == 1:
#         N = t.shape
#     elif len(t.shape) == 2:
#         N, C = t.shape
#     elif len(t.shape) == 3:
#         N, C, M = t.shape
#     elif len(t.shape) == 4:
#         N, C, H, W = t.shape
#     else:
#         raise f'invalid shape {t.shape}'

#     # compute cosine similarity for t and s
#     t = F.cosine_similarity(t.view(N, -1), s.view(N, -1), dim=0).mean()
#     return f * tensor(5).to(f.device) * t

# @register_weight
# def w25_teacher_student(f: Tensor, t: Tensor = None, s: Tensor = None):
#     """w25_teacher_student = f * 25 * t * s"""
#     if t is None and s is None:
#         return f * tensor(25).to(f.device)
#     if len(t.shape) == 1:
#         N = t.shape
#     elif len(t.shape) == 2:
#         N, C = t.shape
#     elif len(t.shape) == 3:
#         N, C, M = t.shape
#     elif len(t.shape) == 4:
#         N, C, H, W = t.shape
#     else:
#         raise f'invalid shape {t.shape}'

#     # compute cosine similarity for t and s
#     t = F.cosine_similarity(t.view(N, -1), s.view(N, -1), dim=0).mean()
#     return f * tensor(25).to(f.device) * t

# @register_weight
# def w50_teacher_student(f: Tensor, t: Tensor = None, s: Tensor = None):
#     """w50_teacher_student = f * 50 * t * s"""
#     if t is None and s is None:
#         return f * tensor(50).to(f.device)
#     if len(t.shape) == 1:
#         N = t.shape
#     elif len(t.shape) == 2:
#         N, C = t.shape
#     elif len(t.shape) == 3:
#         N, C, M = t.shape
#     elif len(t.shape) == 4:
#         N, C, H, W = t.shape
#     else:
#         raise f'invalid shape {t.shape}'

#     # compute cosine similarity for t and s
#     t = F.cosine_similarity(t.view(N, -1), s.view(N, -1), dim=0).mean()
#     return f * tensor(50).to(f.device) * t


@register_weight
def w100_teacher_student(f: Tensor, t: Tensor = None, s: Tensor = None):
    """w100_teacher_student = f * 100 * t * s"""
    if t is None and s is None:
        return f * tensor(100).to(f.device)

    if len(t.shape) == 1:
        N = t.shape
    elif len(t.shape) == 2:
        N, C = t.shape
    elif len(t.shape) == 3:
        N, C, M = t.shape
    elif len(t.shape) == 4:
        N, C, H, W = t.shape
    else:
        raise f'invalid shape {t.shape}'

    # compute cosine similarity for t and s
    t = F.cosine_similarity(t.view(N, -1), s.view(N, -1), dim=0).mean()
    return f * tensor(100).to(f.device) * t
