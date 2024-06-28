from typing import Any, List, Tuple
import pandas as pd
from collections import defaultdict
from surprise import accuracy
from zenml import step

@step
def evaluate_model(svd_model: Any, knn_model: Any, baseline_model: Any, test_data: pd.Series):
    def precision_recall_at_k(predictions, k=10, threshold=3.5):
        user_est_true = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))

        precisions = dict()
        recalls = dict()

        for uid, user_ratings in user_est_true.items():
            user_ratings.sort(key=lambda x: x[0], reverse=True)
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
            n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:k])
            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

        precision = sum(prec for prec in precisions.values()) / len(precisions)
        recall = sum(rec for rec in recalls.values()) / len(recalls)
        return precision, recall

    # Convert test_data back to list of tuples
    test_data_tuples = [(d['userId'], d['movieId'], d['rating']) for d in test_data]

    # Evaluate SVD model
    svd_predictions = svd_model.test(test_data_tuples)
    svd_rmse = accuracy.rmse(svd_predictions, verbose=False)
    svd_mae = accuracy.mae(svd_predictions, verbose=False)
    svd_precision, svd_recall = precision_recall_at_k(svd_predictions)

    print(f'SVD Model - RMSE: {svd_rmse}, MAE: {svd_mae}, Precision at K: {svd_precision}, Recall at K: {svd_recall}')

    # Evaluate KNN model
    knn_predictions = knn_model.test(test_data_tuples)
    knn_rmse = accuracy.rmse(knn_predictions, verbose=False)
    knn_mae = accuracy.mae(knn_predictions, verbose=False)
    knn_precision, knn_recall = precision_recall_at_k(knn_predictions)

    print(f'KNN Model - RMSE: {knn_rmse}, MAE: {knn_mae}, Precision at K: {knn_precision}, Recall at K: {knn_recall}')

    # Evaluate Baseline model
    baseline_predictions = baseline_model.test(test_data_tuples)
    baseline_rmse = accuracy.rmse(baseline_predictions, verbose=False)
    baseline_mae = accuracy.mae(baseline_predictions, verbose=False)
    baseline_precision, baseline_recall = precision_recall_at_k(baseline_predictions)

    print(f'Baseline Model - RMSE: {baseline_rmse}, MAE: {baseline_mae}, Precision at K: {baseline_precision}, Recall at K: {baseline_recall}')
