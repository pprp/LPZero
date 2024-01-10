import numpy as np
from sklearn.feature_selection import mutual_info_regression


def measure_mutual_information(var1, var2, random_state=42):
    """
    Calculate the average mutual information between two variables.

    Parameters:
    - var1 (array-like): The first variable.
    - var2 (array-like): The second variable.
    - random_state (int): A seed used by the random number generator for reproducibility.

    Returns:
    - float: The average mutual information between the two variables.
    """
    # Combine the variables into a single dataset
    X = np.column_stack((var1, var2))

    # Calculate mutual information; since it's between two variables, we don't need to specify labels
    mi = mutual_info_regression(
        X, var2, discrete_features='auto', random_state=random_state
    )

    # Return the average mutual information
    return np.mean(mi)
