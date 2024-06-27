from zenml import step
from surprise import SVD, KNNBasic, BaselineOnly
from surprise.model_selection import GridSearchCV
from typing import Tuple

@step
def hp_tuner(train_data) -> Tuple[dict, dict, dict]:
    # Define hyperparameter grids
    param_grid_svd = {
        'n_factors': [20, 50, 100],
        'lr_all': [0.002, 0.005, 0.01],
        'reg_all': [0.02, 0.05, 0.1]
    }

    param_grid_knn = {
        'k': [20, 40, 60],
        'min_k': [1, 3, 5],
        'sim_options': {
            'name': ['msd', 'cosine', 'pearson'],
            'user_based': [False, True]
        }
    }

    param_grid_baseline = {
        'bsl_options': {
            'method': ['als', 'sgd'],
            'n_epochs': [5, 10, 20],
            'reg_u': [12, 15, 18],
            'reg_i': [5, 10, 15]
        }
    }

    # Perform grid search for each algorithm
    gs_svd = GridSearchCV(SVD, param_grid_svd, measures=['rmse', 'mae'], cv=3)
    gs_svd.fit(train_data)

    gs_knn = GridSearchCV(KNNBasic, param_grid_knn, measures=['rmse', 'mae'], cv=3)
    gs_knn.fit(train_data)

    gs_baseline = GridSearchCV(BaselineOnly, param_grid_baseline, measures=['rmse', 'mae'], cv=3)
    gs_baseline.fit(train_data)

    # Get best parameters
    best_params_svd = gs_svd.best_params['rmse']
    best_params_knn = gs_knn.best_params['rmse']
    best_params_baseline = gs_baseline.best_params['rmse']

    return best_params_svd, best_params_knn, best_params_baseline
