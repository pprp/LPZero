import numpy as np 
import torch 

# Covariance calculations for Jacobian covariance and variations
def covariance(jacobs):
    jacob = torch.transpose(jacobs, 0, 1).reshape(jacobs.size(1), -1).cpu().numpy()
    correlations = np.corrcoef(jacob)
    v, _ = np.linalg.eig(correlations)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1.0 / (v + k))