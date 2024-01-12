import random
from typing import TypeVar, Union

import torch
import torch.nn.functional as F

Scalar = TypeVar('Scalar', torch.Tensor, float, int)
Vector = TypeVar('Vector', torch.Tensor)
Matrix = TypeVar('Matrix', torch.Tensor)

ALLTYPE = Union[Scalar, Vector, Matrix]

UNARY_KEYS = (
    'element_wise_log',
    'element_wise_abslog',
    'element_wise_abs',
    'element_wise_pow',
    'element_wise_exp',
    'normalize',
    'element_wise_relu',
    'element_wise_invert',
    'frobenius_norm',
    'element_wise_normalized_sum',
    'l1_norm',
    'softmax',
    'sigmoid',
    'logsoftmax',
    'element_wise_sqrt',
    'min_max_normalize',
    'to_mean_scalar',
    'to_std_scalar',
    'no_op',
)


# sample key by probability
def sample_unary_key_by_prob(probability=None):
    if probability is None:
        probability = [0.1] * (len(UNARY_KEYS) - 1) + [0.2]
    return random.choices(list(range(len(UNARY_KEYS))), weights=probability, k=1)[0]


# unary operation
def no_op(A: ALLTYPE) -> ALLTYPE:
    return A


def element_wise_log(A: ALLTYPE) -> ALLTYPE:
    A[A <= 0] = 1
    return torch.log(A)


def element_wise_revert(A: ALLTYPE) -> ALLTYPE:
    return A * -1


def element_wise_mish(A: ALLTYPE) -> ALLTYPE:
    return F.mish(A)


def element_wise_swish(A: ALLTYPE) -> ALLTYPE:
    return A * torch.sigmoid(A)


def element_wise_leaky_relu(A: ALLTYPE) -> ALLTYPE:
    return F.leaky_relu(A, negative_slope=0.1)


def min_max_normalize(A: ALLTYPE) -> ALLTYPE:
    A_min, A_max = A.min(), A.max()
    return (A - A_min) / (A_max - A_min + 1e-9)


def element_wise_abslog(A: ALLTYPE) -> ALLTYPE:
    A[A == 0] = 1
    A = torch.abs(A)
    return torch.log(A)


def element_wise_abs(A: ALLTYPE) -> ALLTYPE:
    return torch.abs(A)


def element_wise_sqrt(A: ALLTYPE) -> ALLTYPE:
    A[A <= 0] = 0
    return torch.sqrt(A)


def element_wise_pow(A: ALLTYPE) -> ALLTYPE:
    return torch.pow(A, 2)


def element_wise_exp(A: ALLTYPE) -> ALLTYPE:
    return torch.exp(A)


def normalize(A: ALLTYPE) -> ALLTYPE:
    m = torch.mean(A)
    s = torch.std(A) + 1e-9
    return (A - m) / s


def element_wise_relu(A: ALLTYPE) -> ALLTYPE:
    return F.relu(A)


def element_wise_invert(A: ALLTYPE) -> ALLTYPE:
    if torch.any(A == 0):
        raise ZeroDivisionError
    return 1 / A


def frobenius_norm(A: ALLTYPE) -> Scalar:
    return torch.norm(A, p='fro')


def element_wise_normalized_sum(A: ALLTYPE) -> Scalar:
    return torch.sum(A) / A.numel()


def l1_norm(A: ALLTYPE) -> Scalar:
    return torch.sum(torch.abs(A)) / A.numel()


def p_dist(A: Matrix) -> Vector:
    return F.pdist(A)


def softmax(A: ALLTYPE) -> ALLTYPE:
    return F.softmax(A, dim=0)


def logsoftmax(A: ALLTYPE) -> ALLTYPE:
    return F.log_softmax(A, dim=0)


def sigmoid(A: ALLTYPE) -> ALLTYPE:
    return torch.sigmoid(A)


def min_max_normalize(A: ALLTYPE) -> ALLTYPE:
    A_min, A_max = A.min(), A.max()
    return (A - A_min) / (A_max - A_min + 1e-9)


def to_mean_scalar(A: ALLTYPE) -> Scalar:
    return torch.mean(A)


def to_sum_scalar(A: ALLTYPE) -> Scalar:
    return torch.sum(A)


def to_std_scalar(A: ALLTYPE) -> Scalar:
    return torch.std(A)


def to_sqrt_scalar(A: ALLTYPE) -> Scalar:
    A[A <= 0] = 0
    return torch.sqrt(A)


def gram_matrix(A: Matrix) -> Matrix:
    """https://pytorch.org/tutorials/advanced/neural_style_tutorial.html"""
    assert len(A.shape) == 4, 'Input shape is invalid.'
    a, b, c, d = A.size()
    feature = A.view(a * b, c * d)
    G = torch.mm(feature, feature.t())
    return G.div(a * b * c * d)


def unary_operation(A, idx=None):
    if idx is None:
        idx = random.choice(range(len(UNARY_KEYS)))

    assert idx < len(UNARY_KEYS)

    unaries = {
        'element_wise_log': element_wise_log,
        'element_wise_abslog': element_wise_abslog,
        'element_wise_abs': element_wise_abs,
        'element_wise_pow': element_wise_pow,
        'element_wise_exp': element_wise_exp,
        'normalize': normalize,
        'element_wise_relu': element_wise_relu,
        'element_wise_invert': element_wise_invert,
        'frobenius_norm': frobenius_norm,
        'element_wise_normalized_sum': element_wise_normalized_sum,
        'l1_norm': l1_norm,
        'softmax': softmax,
        'logsoftmax': logsoftmax,
        'sigmoid': sigmoid,
        'min_max_normalize': min_max_normalize,
        'element_wise_sqrt': element_wise_sqrt,
        'to_mean_scalar': to_mean_scalar,
        'to_std_scalar': to_std_scalar,
        'no_op': no_op,
        'p_dist': p_dist,
        'gram_matrix': gram_matrix,
        'element_wise_revert': element_wise_revert,
        'element_wise_mish': element_wise_mish,
        'element_wise_swish': element_wise_swish,
        'element_wise_leaky_relu': element_wise_leaky_relu,
        'to_sqrt_scalar': to_sqrt_scalar,
    }

    return unaries[UNARY_KEYS[idx]](A)
