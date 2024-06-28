from zenml import step
from surprise import Dataset, KNNBasic, SVD, BaselineOnly
from surprise.model_selection import GridSearchCV
from typing import Tuple, Dict

@step
def hp_tuner(dataset: Dataset) -> Tuple[Dict, Dict, Dict]:
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

    gs_svd = GridSearchCV(SVD, param_grid_svd, measures=['rmse'], cv=3)
    gs_knn = GridSearchCV(KNNBasic, param_grid_knn, measures=['rmse'], cv=3)
    gs_baseline = GridSearchCV(BaselineOnly, param_grid_baseline, measures=['rmse'], cv=3)

    gs_svd.fit(dataset)
    gs_knn.fit(dataset)
    gs_baseline.fit(dataset)

    best_params_svd = gs_svd.best_params['rmse']
    best_params_knn = gs_knn.best_params['rmse']
    best_params_baseline = gs_baseline.best_params['rmse']

    return best_params_svd, best_params_knn, best_params_baseline
