import numpy as np
import torch
from sklearn.metrics import pairwise_distances
from . import measure


# Covariance calculations for Jacobian covariance and variations
def covariance(jacobs):
    jacob = torch.transpose(jacobs, 0, 1).reshape(
        jacobs.size(1), -1).cpu().numpy()
    correlations = np.corrcoef(jacob)
    v, _ = np.linalg.eig(correlations)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1.0 / (v + k))


# Cosine calculations for Jacobian cosine and variations
def cosine(jacobs):
    jacob = torch.transpose(jacobs, 0, 1).reshape(
        jacobs.size(1), -1).cpu().numpy()
    norm = np.linalg.norm(jacob, axis=1)
    normed = jacob / norm[:, None]
    cosines = (-pairwise_distances(normed, metric='cosine') + 1) - np.identity(
        normed.shape[0]
    )
    summed = np.sum(np.power(np.absolute(cosines.flatten()), 1.0 / 20)) / 2
    return 1 - (1 / (pow(cosines.shape[0], 2) - cosines.shape[0]) * summed)
