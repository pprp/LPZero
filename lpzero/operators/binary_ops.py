import random
from typing import TypeVar, Union

import torch
import torch.nn.functional as F

Scalar = TypeVar('Scalar')
Vector = TypeVar('Vector')
Matrix = TypeVar('Matrix')

ALLTYPE = Union[Union[Scalar, Vector], Matrix]

BINARY_KEYS = (
    'element_wise_sum',
    'element_wise_difference',
    'element_wise_product',
    'matrix_multiplication',
)

# sample key by probability


def sample_binary_key_by_prob(probability=None):
    if probability is None:
        # the probability from large to small
        probability = (0.6, 0.3, 0.05, 0.05)
    res = random.choices(list(range(len(BINARY_KEYS))),
                         weights=probability, k=1)[0]
    return res


# binary operator


def element_wise_sum(A: ALLTYPE, B: ALLTYPE) -> ALLTYPE:
    return A + B


def element_wise_mean(A: ALLTYPE, B: ALLTYPE) -> ALLTYPE:
    return (A + B) / 2


# SCALAR_DIFF_OP


def element_wise_difference(A: ALLTYPE, B: ALLTYPE) -> ALLTYPE:
    return A - B


def element_wise_product(A: ALLTYPE, B: ALLTYPE) -> ALLTYPE:
    return A * B


def matrix_multiplication(A: Matrix, B: Matrix):
    return A @ B


def lesser_than(A: ALLTYPE, B: ALLTYPE) -> bool:
    return (A < B).float()


def greater_than(A: ALLTYPE, B: ALLTYPE) -> bool:
    return (A > B).float()


def equal_to(A: ALLTYPE, B: ALLTYPE) -> bool:
    return (A == B).float()


def hamming_distance(A: ALLTYPE, B: ALLTYPE) -> Scalar:
    value = torch.tensor([0], dtype=A.dtype)
    A = torch.heaviside(A, values=value)
    B = torch.heaviside(B, values=value)
    return sum(A != B)


def pairwise_distance(A: Matrix, B: Matrix) -> Vector:
    return F.pairwise_distance(A, B, p=2)


def kl_divergence(A: ALLTYPE, B: ALLTYPE) -> Scalar:
    return torch.nn.KLDivLoss('batchmean')(A, B)


def cosine_similarity(A: Matrix, B: Matrix) -> Scalar:
    A = A.reshape(A.shape[0], -1)
    B = A.reshape(B.shape[0], -1)
    C = torch.nn.CosineSimilarity()(A, B)
    return torch.sum(C)


def mse_loss(A: ALLTYPE, B: ALLTYPE) -> Scalar:
    return F.mse_loss(A, B)


def l1_loss(A: ALLTYPE, B: ALLTYPE) -> Scalar:
    return F.l1_loss(A, B)


def binary_operation(A, B, idx=None):
    # 10
    if idx is None:
        idx = random.choice(range(len(BINARY_KEYS)))
    
    if A is None and B is None:
        raise ValueError('A and B cannot be both None')
    if A is None:
        return B
    if B is None:
        return A

    assert idx < len(BINARY_KEYS)

    binaries = {
        'element_wise_sum': element_wise_sum,
        'element_wise_difference': element_wise_difference,
        'element_wise_product': element_wise_product,
        'matrix_multiplication': matrix_multiplication,
        'lesser_than': lesser_than,
        'greater_than': greater_than,
        'equal_to': equal_to,
        'hamming_distance': hamming_distance,
        'kl_divergence': kl_divergence,
        'cosine_similarity': cosine_similarity,
        'pairwise_distance': pairwise_distance,
        'l1_loss': l1_loss,
        'mse_loss': mse_loss,
    }
    return binaries[BINARY_KEYS[idx]](A, B)
