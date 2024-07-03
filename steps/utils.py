import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np

def compute_similarity_matrix(features: pd.DataFrame, similarity_measure: str, n_components: int) -> np.ndarray:
    """
    Compute the similarity matrix for numerical features using a specified similarity measure and PCA for dimensionality reduction.

    Args:
        features (pd.DataFrame): DataFrame containing numerical features.
        similarity_measure (str): The similarity measure to use ('cosine' or 'euclidean').
        n_components (int): Number of PCA components for dimensionality reduction.

    Returns:
        np.ndarray: The similarity matrix.
    """
    # Convert to float32
    features = features.astype(np.float32)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features)

    # Compute similarity matrix
    if similarity_measure == 'cosine':
        similarity_matrix = cosine_similarity(features_pca)
    elif similarity_measure == 'euclidean':
        similarity_matrix = euclidean_distances(features_pca)
    else:
        raise ValueError(f"Unknown similarity measure: {similarity_measure}")

    return similarity_matrix
