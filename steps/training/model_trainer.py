from zenml import step
from surprise import SVD, KNNBasic, BaselineOnly
from typing import Tuple

@step
def model_trainer(train_data, best_params_svd: dict, best_params_knn: dict, best_params_baseline: dict) -> Tuple[SVD, KNNBasic, BaselineOnly]:
    # Train SVD model
    svd = SVD(**best_params_svd)
    svd.fit(train_data)

    # Train KNN model
    knn = KNNBasic(**best_params_knn)
    knn.fit(train_data)

    # Train Baseline model
    baseline = BaselineOnly(**best_params_baseline)
    baseline.fit(train_data)

    return svd, knn, baseline
