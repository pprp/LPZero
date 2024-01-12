import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def measure_silhouette(var1, var2, n_clusters=8, random_state=42):
    """
    Calculate the silhouette score for the clustering of two variables.

    Parameters:
    - var1 (array-like): The first variable (array-like).
    - var2 (array-like): The second variable (array-like).
    - n_clusters (int): The number of clusters to use.
    - random_state (int): A seed used by the random number generator for reproducibility.

    Returns:
    - float: The average silhouette score.
    """
    # Combine the variables into a single dataset
    X = np.column_stack((var1, var2))

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=random_state, n_init='auto')
    labels = kmeans.fit_predict(X)

    # Calculate the silhouette score
    silhouette_avg = silhouette_score(X, labels)

    return silhouette_avg
