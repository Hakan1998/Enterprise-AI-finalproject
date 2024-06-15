from zenml import step
import pandas as pd
import numpy as np
from typing import Any

@step
def predictor(service: Any, input_data: pd.DataFrame, model_type: str) -> np.ndarray:
    """
    Makes predictions using a trained model.

    Args:
        service (Any): The trained model.
        input_data (pd.DataFrame): The input data for inference.
        model_type (str): The type of model being used ("baseline", "collaborative", "content_based", "matrix_factorization").

    Returns:
        np.ndarray: The model predictions.
    """
    if model_type == "baseline":
        predictions = np.array([service] * len(input_data))
    elif model_type == "collaborative":
        user_item_matrix = input_data.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
        distances, indices = service.kneighbors(user_item_matrix, n_neighbors=1)
        predictions = np.array([user_item_matrix.iloc[ind[0]].mean() for ind in indices])
    elif model_type == "content_based":
        similarity_matrix = service
        predictions = similarity_matrix.dot(input_data.T).diagonal()
    elif model_type == "matrix_factorization":
        user_item_matrix = input_data.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
        W = service.transform(user_item_matrix)
        H = service.components_
        predictions = np.dot(W, H).diagonal()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return predictions