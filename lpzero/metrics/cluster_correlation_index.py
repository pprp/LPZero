import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import silhouette_score


def measure_cluster_corr_index(
    var1, var2, max_mutual_information=1, n_clusters=3, lam=5, SEEDED_RANDOM_STATE=42
):
    """
    Evaluate the distinguishability of two variables using various methods.

    Parameters:
    - X (numpy.ndarray): A 2D array where each row is an observation and
      the two columns represent the two variables.
    - n_clusters (int): Number of clusters to use for silhouette score.

    Returns:
    - dict: A dictionary containing the results of the various methods.
    """
    X = np.column_stack((var1, var2))

    # Ensure correct shape for mutual information calculation
    if X.shape[1] != 2:
        raise ValueError(
            'Input array X must have exactly two columns representing the two variables.'
        )

    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=SEEDED_RANDOM_STATE).fit(X)
    labels = kmeans.labels_

    # Mutual Information
    mi = mutual_info_regression(
        X, labels, discrete_features='auto', random_state=SEEDED_RANDOM_STATE
    )
    mi = np.mean(mi) / max_mutual_information

    # Spearman Correlation
    spearman_corr, _ = spearmanr(X[:, 0], X[:, 1])

    return spearman_corr + lam * mi 
