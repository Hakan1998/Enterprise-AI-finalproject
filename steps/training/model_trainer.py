from zenml import step
from surprise import SVD, KNNBasic, BaselineOnly, Trainset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from typing import Tuple, Dict, Any
import pandas as pd

@step(enable_cache=False)
def model_trainer(
    train_data: Trainset, 
    raw_train_data: pd.DataFrame, 
    best_params_svd: Dict[str, Any], 
    best_params_knn: Dict[str, Any], 
    best_params_baseline: Dict[str, Any], 
    best_params_content: Dict[str, Any]
) -> Tuple[SVD, KNNBasic, BaselineOnly, Dict[str, Any]]:
    """
    Train multiple models (SVD, KNN, Baseline, Content-based) using the best hyperparameters and return the trained models.

    Args:
        train_data (Trainset): The Surprise trainset object used for training collaborative filtering models.
        raw_train_data (pd.DataFrame): The raw training data containing 'userId', 'movieId', 'rating', and 'genres' columns.
        best_params_svd (Dict[str, Any]): Best hyperparameters for the SVD model.
        best_params_knn (Dict[str, Any]): Best hyperparameters for the KNN model.
        best_params_baseline (Dict[str, Any]): Best hyperparameters for the Baseline model.
        best_params_content (Dict[str, Any]): Best hyperparameters for the content-based model.

    Returns:
        Tuple[SVD, KNNBasic, BaselineOnly, Dict[str, Any]]:
            - SVD: The trained SVD model.
            - KNNBasic: The trained KNN model.
            - BaselineOnly: The trained Baseline model.
            - Dict[str, Any]: The trained content-based model containing the TF-IDF matrix and cosine similarity matrix.
    """
    # Train Collaborative Filtering Models
    svd = SVD(**best_params_svd)
    knn = KNNBasic(**best_params_knn)
    baseline = BaselineOnly(**best_params_baseline)

    svd.fit(train_data)
    knn.fit(train_data)
    baseline.fit(train_data)

    # Ensure that ngram_range is passed as a tuple
    best_params_content['ngram_range'] = tuple(best_params_content['ngram_range'])

    # Train Content-based Filtering using TF-IDF on genres with best hyperparameters
    tfidf = TfidfVectorizer(stop_words='english', **best_params_content)
    tfidf_matrix = tfidf.fit_transform(raw_train_data['genres'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    content_model = {
        'tfidf_matrix': tfidf_matrix,
        'cosine_sim': cosine_sim
    }

    return svd, knn, baseline, content_model
