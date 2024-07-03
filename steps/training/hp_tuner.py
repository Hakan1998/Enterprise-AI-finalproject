from zenml import step
from surprise import Dataset, KNNBasic, SVD, BaselineOnly
from surprise.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, Dict, Any
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel

def perform_grid_search(model_class: Any, param_grid: Dict[str, Any], dataset: Dataset) -> Dict[str, Any]:
    """
    Perform grid search for a given model class and parameter grid.

    Args:
        model_class (Any): The model class to perform grid search on.
        param_grid (Dict[str, Any]): The parameter grid for grid search.
        dataset (Dataset): The dataset for grid search.

    Returns:
        Dict[str, Any]: The best parameters found in grid search.
    """
    grid_search = GridSearchCV(model_class, param_grid, measures=['rmse'], cv=3)
    grid_search.fit(dataset)
    return grid_search.best_params['rmse']

def tune_content_based(raw_train_data: pd.DataFrame, param_grid: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tune hyperparameters for content-based filtering using TF-IDF.

    Args:
        raw_train_data (pd.DataFrame): The raw training data.
        param_grid (Dict[str, Any]): The parameter grid for TF-IDF.

    Returns:
        Dict[str, Any]: The best parameters found for the content-based model.
    """
    best_score = float('inf')
    best_params = None
    for max_df in param_grid['max_df']:
        for min_df in param_grid['min_df']:
            for ngram_range in param_grid['ngram_range']:
                tfidf = TfidfVectorizer(stop_words='english', max_df=max_df, min_df=min_df, ngram_range=ngram_range)
                try:
                    tfidf_matrix = tfidf.fit_transform(raw_train_data['budget'])
                    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

                    # Calculate score (e.g., average similarity) for these parameters
                    avg_sim_score = cosine_sim.mean()
                    if avg_sim_score < best_score:
                        best_score = avg_sim_score
                        best_params = {
                            'max_df': max_df,
                            'min_df': min_df,
                            'ngram_range': ngram_range
                        }
                except ValueError as e:
                    print(f"Skipping parameters max_df={max_df}, min_df={min_df}, ngram_range={ngram_range} due to error: {e}")
                    continue
    return best_params

@step(enable_cache=False)
def hp_tuner(dataset: Dataset, raw_train_data: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Hyperparameter tuner step for multiple models (SVD, KNN, Baseline, Content-based).

    Args:
        dataset (Dataset): The dataset for collaborative filtering models.
        raw_train_data (pd.DataFrame): The raw training data for content-based model.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]: 
        Best parameters for SVD, KNN, Baseline, and content-based models.
    """
    # Define hyperparameter grids
    param_grid_svd = {
        'n_epochs': [20, 30],
        'lr_all': [0.002, 0.005],
        'reg_all': [0.4, 0.6]
    }
    param_grid_knn = {
        'k': [20, 30],
        'sim_options': {
            'name': ['msd', 'cosine'],
            'user_based': [False]
        }
    }
    param_grid_baseline = {
        'bsl_options': {
            'method': ['als', 'sgd'],
            'n_epochs': [5, 10]
        }
    }
    param_grid_content = {
        'max_df': [0.8, 1.0],  # Ensure terms are kept by increasing max_df
        'min_df': [0.0, 0.1],  # Ensure terms are kept by decreasing min_df
        'ngram_range': [(1, 1), (1, 2)]  # Use tuples instead of lists
    }

    # Perform grid search for collaborative filtering algorithms
    best_params_svd = perform_grid_search(SVD, param_grid_svd, dataset)
    best_params_knn = perform_grid_search(KNNBasic, param_grid_knn, dataset)
    best_params_baseline = perform_grid_search(BaselineOnly, param_grid_baseline, dataset)

    # Tune hyperparameters for content-based model
    best_params_content = tune_content_based(raw_train_data, param_grid_content)

    return best_params_svd, best_params_knn, best_params_baseline, best_params_content