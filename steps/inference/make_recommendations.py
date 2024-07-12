from typing import List, Dict, Any, Tuple
from surprise import Trainset
import pandas as pd
from zenml import step
from surprise import Dataset, Reader
from surprise import AlgoBase
from typing import Annotated


def convert_to_surprise_format(preprocessed_data: pd.DataFrame) -> Dataset:
    """
    Convert preprocessed data to the format required by the Surprise library.

    Parameters:
    preprocessed_data (pd.DataFrame): The preprocessed data containing userId, itemId, and rating columns.

    Returns:
    Dataset: The Surprise Dataset.
    """
    reader = Reader(rating_scale=(1, 5))
    return Dataset.load_from_df(preprocessed_data[['userId', 'id', 'rating']], reader)

@step
def make_recommendations(model: AlgoBase, raw_test_data: pd.DataFrame, k: int = 10) -> Annotated[pd.DataFrame, "User Movie Recommendation"]:
    """
    Based on out referance Dataset this functions creates movie Recommendations for each User on Movies they havent seen(rated) yet.

    Parameters:
    model (AlgoBase): The trained recommendation model.
    raw_test_data (pd.DataFrame): The dataset containing userId, id, and rating columns, plus any additional columns.
    k (int): The number of top recommendations to generate for each user.

    Returns:
    pd.DataFrame: A DataFrame containing userId and recommended movie ids.
    """
    # Ensure raw_test_data is a DataFrame
    raw_test_data = raw_test_data.load() if hasattr(raw_test_data, 'load') else raw_test_data

    # Get list of unique users
    unique_users = raw_test_data['userId'].unique()

    recommendations = []

    for uid in unique_users:
        # Get items already rated by the user
        rated_items = set(raw_test_data[raw_test_data['userId'] == uid]['id'])
        
        # Generate predictions for items not rated by the user
        user_recommendations = []
        for iid in raw_test_data['id'].unique():
            if iid not in rated_items:
                prediction = model.predict(str(uid), str(iid))
                user_recommendations.append((iid, prediction.est))
        
        # Sort recommendations for the user and select top K
        user_recommendations.sort(key=lambda x: x[1], reverse=True)
        top_k_recommendations = user_recommendations[:k]

        for iid, predicted_rating in top_k_recommendations:
            recommendations.append({'userId': uid, 'id': iid, 'predicted_rating': predicted_rating})

    # Convert recommendations to DataFrame
    recommendations_df = pd.DataFrame(recommendations)
    print(recommendations_df)
    return recommendations_df