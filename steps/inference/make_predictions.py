from typing import List, Dict, Any
from surprise import Trainset
import pandas as pd
from zenml import step

@step
def make_predictions(model: Any, test_data: pd.Series, k: int = 10) -> Dict[int, List[int]]:
    """
    Make top k predictions using the trained model on the test data.

    Args:
        model: The trained Surprise model.
        test_data (pd.Series): The test data as a series of dictionaries.
        k (int): Number of top recommendations to return per user.

    Returns:
        Dict[int, List[int]]: A dictionary where keys are userIds and values are lists of top k recommended movieIds.
    """
    user_ids = test_data.apply(lambda x: x['userId']).unique()
    all_movie_ids = test_data.apply(lambda x: x['id']).unique()

    top_k_recommendations = {}
    
    for user_id in user_ids:
        user_predictions = [
            (movie_id, model.predict(user_id, movie_id).est)
            for movie_id in all_movie_ids
        ]
        
        top_k_movies = sorted(user_predictions, key=lambda x: x[1], reverse=True)[:k]
        top_k_recommendations[user_id] = [movie_id for movie_id, _ in top_k_movies]

    return top_k_recommendations