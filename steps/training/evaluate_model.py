from zenml import step
import numpy as np
import pandas as pd
from typing import Tuple, Any, List

def precision_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    recommended_at_k = recommended[:k]
    relevant_set = set(relevant)
    recommended_set = set(recommended_at_k)
    relevant_recommended = relevant_set.intersection(recommended_set)
    return len(relevant_recommended) / k

def recall_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    recommended_at_k = recommended[:k]
    relevant_set = set(relevant)
    relevant_recommended = relevant_set.intersection(set(recommended_at_k))
    return len(relevant_recommended) / len(relevant)

def ndcg_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    recommended_at_k = recommended[:k]
    dcg = 0.0
    idcg = 0.0
    for i in range(len(recommended_at_k)):
        if recommended_at_k[i] in relevant:
            dcg += 1.0 / np.log2(i + 2)
    for i in range(min(k, len(relevant))):
        idcg += 1.0 / np.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0

@step
def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_type: str) -> Tuple[float, float, float]:
    """
    Evaluate the performance of a model on the test data.

    Args:
        model (Any): The trained model.
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The test labels.
        model_type (str): The type of model being evaluated ("baseline", "collaborative", "content_based", "matrix_factorization").

    Returns:
        Tuple[float, float, float]: The precision@K, recall@K, and ndcg@K of the model predictions.
    """
    # Placeholder for predictions based on the model type
    if model_type == "baseline":
        y_pred = [model] * len(X_test)
    elif model_type == "collaborative":
        user_item_matrix = X_test.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
        distances, indices = model.kneighbors(user_item_matrix, n_neighbors=10)
        y_pred = [list(user_item_matrix.columns[indices[i]]) for i in range(len(indices))]
    elif model_type == "content_based":
        similarity_matrix = model
        y_pred = similarity_matrix.dot(X_test.T).argsort(axis=1)[:, -10:][:, ::-1]
    elif model_type == "matrix_factorization":
        user_item_matrix = X_test.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
        W = model.transform(user_item_matrix)
        H = model.components_
        y_pred_matrix = np.dot(W, H)
        y_pred = np.argsort(y_pred_matrix, axis=1)[:, -10:][:, ::-1]
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Calculate precision, recall, and ndcg at k=10
    precision = np.mean([precision_at_k(y_pred[i], y_test.iloc[i], 10) for i in range(len(y_test))])
    recall = np.mean([recall_at_k(y_pred[i], y_test.iloc[i], 10) for i in range(len(y_test))])
    ndcg = np.mean([ndcg_at_k(y_pred[i], y_test.iloc[i], 10) for i in range(len(y_test))])

    return precision, recall, ndcg