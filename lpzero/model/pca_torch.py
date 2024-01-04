import numpy as np
import torch


def cov(tensor, rowvar=True, bias=False):
    """Estimate a covariance matrix (np.cov)
    https://gist.github.com/ModarTensai/5ab449acba9df1a26c12060240773110
    """
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()


@torch.no_grad()
def pca_torch(data, energy):
    # retuerns eig vals and vecs as well as the number of pc's needed to capture proportions of variance
    data = data - data.mean(0)
    covariance = cov(data, rowvar=False)
    # covariance = np.nan_to_num(covariance)  # I added this to fix one problem that one matrix was causing in one opt
    eigenvalues, eigenvectors = torch.linalg.eigh(
        covariance
    )  # eigenvals are sorted small to large

    en_evecs = np.zeros(len(energy))
    total = torch.sum(eigenvalues)
    eigenvalues = eigenvalues.cpu().numpy()
    for en, idx_en in zip(energy, range(len(energy))):
        accum = 0
        k = 1
        while accum <= en:
            accum += eigenvalues[-k] / total
            k += 1
        en_evecs[idx_en] = (
            k - 1
        )  # en_evecs is num of eigenvectors needed to explain en proportion of variance
    return en_evecs  # , eigenvalues, eigenvectors


def pca(data, energy):
    # retuerns eig vals and vecs as well as the number of pc's needed to capture proportions of variance
    data = data - data.mean(axis=0)
    covariance = np.cov(data, rowvar=False)
    covariance = np.nan_to_num(
        covariance
    )  # I added this to fix one problem that one matrix was causing in one opt
    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )  # eigenvals are sorted small to large

    en_evecs = np.zeros(len(energy))
    total = np.sum(eigenvalues)
    for en, idx_en in zip(energy, range(len(energy))):
        accum = 0
        k = 1
        while accum <= en:
            accum += eigenvalues[-k] / total
            k += 1
        en_evecs[idx_en] = (
            k - 1
        )  # en_evecs is num of eigenvectors needed to explain en proportion of variance
    return en_evecs  # , eigenvalues, eigenvectors
