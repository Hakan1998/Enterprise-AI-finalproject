from zenml import step
from surprise import Dataset, KNNBasic, SVD, BaselineOnly, NormalPredictor, NMF, SlopeOne
from surprise.model_selection import GridSearchCV
from typing import Tuple, Dict, Any, Annotated
import pandas as pd

def perform_grid_search(model_class: Any, param_grid: Dict[str, Any], dataset: Dataset) -> Dict[str, Any]:
    grid_search = GridSearchCV(model_class, param_grid, measures=['rmse'], cv=3)
    grid_search.fit(dataset)
    return grid_search.best_params['rmse']

def tune_content_based(raw_train_data: pd.DataFrame) -> Dict[str, Any]:
    best_params = {
        'max_df': 0.8,
        'min_df': 0.01,
        'ngram_range': (1, 4)  # Ensure this is a tuple
    }
    return best_params

@step(enable_cache=False)
def hp_tuner(
    dataset: Dataset, 
    raw_train_data: pd.DataFrame
) -> Tuple[
    Annotated[Dict[str, Any], "SVD Model"],
    Annotated[Dict[str, Any], "KNN Model"],
    Annotated[Dict[str, Any], "Baseline Only Model"],
    Annotated[Dict[str, Any], "Normal Predictor Model"],
    Annotated[Dict[str, Any], "NMF Model"],
    Annotated[Dict[str, Any], "Slope One Model"],
    Annotated[Dict[str, Any], "Content-based Model"]
]:
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
    param_grid_normal = {
        # NormalPredictor has no hyperparameters to tune
    }
    param_grid_nmf = {
        'n_factors': [15, 30],
        'n_epochs': [20, 50],
        'biased': [False, True]
    }
    param_grid_slope_one = {
        # SlopeOne has no hyperparameters to tune
    }

    best_params_svd = perform_grid_search(SVD, param_grid_svd, dataset)
    best_params_knn = perform_grid_search(KNNBasic, param_grid_knn, dataset)
    best_params_baseline = perform_grid_search(BaselineOnly, param_grid_baseline, dataset)
    best_params_normal = {}  # No parameters to tune for NormalPredictor
    best_params_nmf = perform_grid_search(NMF, param_grid_nmf, dataset)
    best_params_slope_one = {}  # No parameters to tune for SlopeOne
    best_params_content = tune_content_based(raw_train_data)

    return best_params_svd, best_params_knn, best_params_baseline, best_params_normal, best_params_nmf, best_params_slope_one, best_params_content
