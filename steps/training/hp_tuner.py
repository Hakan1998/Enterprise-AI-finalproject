from zenml import step
from surprise import KNNBasic, SVD, Dataset, Reader
from surprise.model_selection import GridSearchCV
import pandas as pd
from typing import Dict, Any

@step
def hp_tuner(train_data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train_data[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()

    # Tuning User CF
    param_grid_user_cf = {
        'k': [10, 20, 30],
        'sim_options': {
            'name': ['cosine', 'msd', 'pearson'],
            'user_based': [True]
        }
    }
    gs_user_cf = GridSearchCV(KNNBasic, param_grid_user_cf, measures=['rmse'], cv=3)
    gs_user_cf.fit(data)
    best_params_user_cf = gs_user_cf.best_params['rmse']

    # Tuning Item CF
    param_grid_item_cf = {
        'k': [10, 20, 30],
        'sim_options': {
            'name': ['cosine', 'msd', 'pearson'],
            'user_based': [False]
        }
    }
    gs_item_cf = GridSearchCV(KNNBasic, param_grid_item_cf, measures=['rmse'], cv=3)
    gs_item_cf.fit(data)
    best_params_item_cf = gs_item_cf.best_params['rmse']

    # Tuning SVD
    param_grid_svd = {
        'n_factors': [50, 100, 150],
        'lr_all': [0.002, 0.005, 0.01],
        'reg_all': [0.02, 0.1, 0.2]
    }
    gs_svd = GridSearchCV(SVD, param_grid_svd, measures=['rmse'], cv=3)
    gs_svd.fit(data)
    best_params_svd = gs_svd.best_params['rmse']

    # Tuning Content-Based
    best_params_content = {
        'stop_words': 'english',
        'ngram_range': (1, 2)
    }

    best_params = {
        'user_cf': best_params_user_cf,
        'item_cf': best_params_item_cf,
        'svd': best_params_svd,
        'content': best_params_content
    }
    
    return best_params
