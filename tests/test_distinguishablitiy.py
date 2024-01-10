import math
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KernelDensity

SEEDED_RANDOM_STATE = random.randint(0, 100)


def sigmoid(x, k=1):
    """Apply a sigmoid function to x with steepness k."""
    return 1 / (1 + math.exp(-k * x))


def log_transform(x, c=10):
    """Apply a logarithmic transformation to x with scaling constant c."""
    return math.log(1 + c * abs(x)) / math.log(1 + c)


def evaluate_distinguishability(X, max_mutual_information=1, n_clusters=3, lam=5):
    """
    Evaluate the distinguishability of two variables using various methods.

    Parameters:
    - X (numpy.ndarray): A 2D array where each row is an observation and
      the two columns represent the two variables.
    - n_clusters (int): Number of clusters to use for silhouette score.

    Returns:
    - dict: A dictionary containing the results of the various methods.
    """

    # Ensure correct shape for mutual information calculation
    if X.shape[1] != 2:
        raise ValueError(
            'Input array X must have exactly two columns representing the two variables.'
        )

    results = {}

    # Clustering Tendency: Silhouette Score
    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=SEEDED_RANDOM_STATE).fit(X)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(X, labels)
    results['silhouette_score'] = silhouette_avg

    # Mutual Information
    mi = mutual_info_regression(
        X, labels, discrete_features='auto', random_state=SEEDED_RANDOM_STATE
    )
    results['mutual_information'] = np.mean(mi)

    # New metric
    normalized_silhouette = (
        results['silhouette_score'] + 1
    ) / 2  # Normalize silhouette to 0-1 scale
    normalized_mutual_information = (
        results['mutual_information'] / max_mutual_information
    )  # Normalize MI to 0-1 scale

    # Combine the two normalized metrics
    # metric = normalized_mi * normalized_silhouette
    metric = metric = (1 - normalized_silhouette) * \
        (1 - normalized_mutual_information)

    results['new_metric'] = metric

    # Spearman Correlation
    spearman_corr, _ = spearmanr(X[:, 0], X[:, 1])
    results['spearman_correlation'] = spearman_corr

    # Combine Spearman Correlation with the new metric
    # Since Spearman ranges from -1 to 1, normalize it to 0 to 1 scale
    results['combined_metric'] = spearman_corr + lam * metric

    return results


def generate_demo_data1(seed=SEEDED_RANDOM_STATE):
    """
    Generates two sets of data for demonstration. One with multiple clusters and one without clear clusters.
    Both sets will have a positive correlation with each other.
    """
    np.random.seed(seed)
    # Variable 1: Multiple clusters
    variable1_cluster1 = np.random.normal(loc=0, scale=0.5, size=50)
    variable1_cluster2 = np.random.normal(loc=5, scale=0.5, size=50)
    variable1_cluster3 = np.random.normal(loc=10, scale=0.5, size=50)
    variable1_values = np.concatenate(
        (variable1_cluster1, variable1_cluster2, variable1_cluster3)
    )

    # Variable 2: No clear clusters, but with positive correlation with variable 1
    variable2_values = variable1_values + \
        np.random.normal(loc=0, scale=2, size=150)

    return variable1_values, variable2_values


def generate_demo_data2():
    np.random.seed(SEEDED_RANDOM_STATE)
    variable1_values = np.random.normal(loc=0, scale=1, size=100)
    variable2_values = variable1_values + \
        np.random.normal(loc=0, scale=1, size=100)
    return variable1_values, variable2_values


# Generate demo data
variable1_values, variable2_values = generate_demo_data1()

# Combine the variables to create a dataset
X_demo2 = np.column_stack((variable1_values, variable2_values))

# Evaluate the distinguishability of the demo data
results_demo2 = evaluate_distinguishability(X_demo2, n_clusters=3)
print(results_demo2)
# Plot the data
plt.figure(figsize=(8, 10))
plt.scatter(X_demo2[:, 0], X_demo2[:, 1], c='blue')
plt.title(
    f'Silhouette: {results_demo2["silhouette_score"]:.2f}\nMutual Information: {results_demo2["mutual_information"]:.2f}\nNew Metric: {results_demo2["new_metric"]:.2f}\nCombined metric: {results_demo2["combined_metric"]:.2f}\nSpearman Correlation: {results_demo2["spearman_correlation"]:.2f}'
)
plt.xlabel('Variable 1')
plt.ylabel('Variable 2')
plt.savefig('demo_data_visualization1.png')
plt.clf()

# Generate demo data 2
variable1_values, variable2_values = generate_demo_data2()

# Combine the variables to create a dataset
X_demo2 = np.column_stack((variable1_values, variable2_values))

# Evaluate the distinguishability of the demo data
results_demo2 = evaluate_distinguishability(X_demo2, n_clusters=3)
print(results_demo2)
# Plot the data
plt.figure(figsize=(8, 10))
plt.scatter(X_demo2[:, 0], X_demo2[:, 1], c='blue')
plt.title(
    f'Silhouette: {results_demo2["silhouette_score"]:.2f}\nMutual Information: {results_demo2["mutual_information"]:.2f}\nNew Metric: {results_demo2["new_metric"]:.2f}\nCombined metric: {results_demo2["combined_metric"]:.2f}\nSpearman Correlation: {results_demo2["spearman_correlation"]:.2f}'
)
plt.xlabel('Variable 1')
plt.ylabel('Variable 2')
plt.savefig('demo_data_visualization2.png')
