from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression
import numpy as np
from scipy.stats import spearmanr


def measure_cluster_corr_index(var1, var2, max_mutual_information=1, n_clusters=3, lam=5,
                                SEEDED_RANDOM_STATE=42):
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
        raise ValueError("Input array X must have exactly two columns representing the two variables.")
    
    results = {}
    
    # Clustering Tendency: Silhouette Score
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEEDED_RANDOM_STATE).fit(X)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(X, labels)
    results['silhouette_score'] = silhouette_avg
    
    # Mutual Information
    mi = mutual_info_regression(X, labels, discrete_features='auto', random_state=SEEDED_RANDOM_STATE)
    results['mutual_information'] = np.mean(mi)
    
    # New metric 
    normalized_silhouette = (results['silhouette_score'] + 1) / 2  # Normalize silhouette to 0-1 scale
    normalized_mutual_information = results['mutual_information'] / max_mutual_information  # Normalize MI to 0-1 scale
    
    # Combine the two normalized metrics
    # metric = normalized_mi * normalized_silhouette
    metric = metric = (1 - normalized_silhouette) * (1 - normalized_mutual_information)

    results['new_metric'] = metric
    
    # Spearman Correlation
    spearman_corr, _ = spearmanr(X[:, 0], X[:, 1])
    results['spearman_correlation'] = spearman_corr
    
    return spearman_corr + lam * metric