from zenml import step
from surprise import SVD, KNNBasic, BaselineOnly, Trainset
from typing import Tuple

@step
def model_trainer(train_data: Trainset, best_params_svd: dict, best_params_knn: dict, best_params_baseline: dict) -> Tuple[SVD, KNNBasic, BaselineOnly]:
    svd = SVD(**best_params_svd)
    knn = KNNBasic(**best_params_knn)
    baseline = BaselineOnly(**best_params_baseline)

    svd.fit(train_data)
    knn.fit(train_data)
    baseline.fit(train_data)

    return svd, knn, baseline