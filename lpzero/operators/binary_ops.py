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
    'hamming_distance',
    'pairwise_distance',
    'kl_divergence',
    'cosine_similarity',
    'mse_loss',
    'l1_loss',
)


# sample key by probability
def sample_binary_key_by_prob(probability=None):
    if probability is None:
        # Equal probability for all operations
        probability = (0.1,) * len(BINARY_KEYS)
    res = random.choices(list(range(len(BINARY_KEYS))),
                         weights=probability, k=1)[0]
    return res


# binary operator
def element_wise_sum(A: ALLTYPE, B: ALLTYPE) -> ALLTYPE:
    return A + B


def element_wise_difference(A: ALLTYPE, B: ALLTYPE) -> ALLTYPE:
    return A - B


def element_wise_product(A: ALLTYPE, B: ALLTYPE) -> ALLTYPE:
    return A * B


def matrix_multiplication(A: Matrix, B: Matrix):
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return A @ B
    else:
        return A * B


def hamming_distance(A: ALLTYPE, B: ALLTYPE) -> Scalar:
    # Assuming A and B are binary tensors
    return torch.sum(A != B)


def pairwise_distance(A: Matrix, B: Matrix) -> Vector:
    return F.pairwise_distance(A, B, p=2)


def kl_divergence(A: ALLTYPE, B: ALLTYPE) -> Scalar:
    # Ensure A is log-probabilities and B is probabilities
    return F.kl_div(A, B, reduction='batchmean')


def cosine_similarity(A: Matrix, B: Matrix) -> Scalar:
    A = A.reshape(A.shape[0], -1)
    B = B.reshape(B.shape[0], -1)
    C = torch.nn.CosineSimilarity()(A, B)
    return torch.sum(C)


def mse_loss(A: ALLTYPE, B: ALLTYPE) -> Scalar:
    return F.mse_loss(A, B)


def l1_loss(A: ALLTYPE, B: ALLTYPE) -> Scalar:
    return F.l1_loss(A, B)


def binary_operation(A, B, idx=None):
    if idx is None:
        idx = random.choice(range(len(BINARY_KEYS)))

    assert idx < len(BINARY_KEYS)

    binaries = {
        'element_wise_sum': element_wise_sum,
        'element_wise_difference': element_wise_difference,
        'element_wise_product': element_wise_product,
        'matrix_multiplication': matrix_multiplication,
        'hamming_distance': hamming_distance,
        'kl_divergence': kl_divergence,
        'cosine_similarity': cosine_similarity,
        'pairwise_distance': pairwise_distance,
        'l1_loss': l1_loss,
        'mse_loss': mse_loss,
    }

    return binaries[BINARY_KEYS[idx]](A, B)
