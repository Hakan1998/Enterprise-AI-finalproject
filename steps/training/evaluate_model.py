from typing import Any, Tuple, Dict, List
import pandas as pd
from collections import defaultdict
from surprise import accuracy
from zenml import step


def precision_recall_at_k(predictions: List[Tuple], k: int = 10, threshold: float = 3.5) -> Tuple[float, float]:
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = min(k, len(user_ratings))  # Adjust n_rec_k to be the minimum of k and the number of user ratings
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:n_rec_k])
        
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    precision = sum(prec for prec in precisions.values()) / len(precisions)
    recall = sum(rec for rec in recalls.values()) / len(recalls)
    return precision, recall

def evaluate_model_predictions(model: Any, test_data: List[Tuple], k: int = 10) -> Tuple[float, float, float, float]:
    predictions = model.test(test_data)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    precision, recall = precision_recall_at_k(predictions, k)
    return rmse, mae, precision, recall

def evaluate_content_based(raw_test_data: pd.DataFrame, cosine_sim: Any, k: int = 10) -> Tuple[float, float]:
    hits = 0
    total_relevant = 0
    total_recommended = 0
    
    users = raw_test_data['userId'].unique()
    
    for user in users:
        user_data = raw_test_data[raw_test_data['userId'] == user]
        
        for _, row in user_data.iterrows():
            movie_id = row['id']
            actual_rating = row['rating']
            
            if movie_id not in raw_test_data['id'].values:
                continue
            
            try:
                idx = raw_test_data.index[raw_test_data['id'] == movie_id][0]
                sim_scores = list(enumerate(cosine_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                sim_scores = sim_scores[1:k+1]
                
                recommended_movie_ids = [raw_test_data.iloc[i[0]]['id'] for i in sim_scores]
                total_recommended += len(recommended_movie_ids)
                
                if actual_rating >= 2.9:  # Assuming ratings of 3 and above are positive
                    total_relevant += 1
                    if movie_id in recommended_movie_ids:
                        hits += 1
            except IndexError:
                continue
    
    precision = float(hits / total_recommended) if total_recommended > 0 else 0.0
    recall = float(hits / total_relevant) if total_relevant > 0 else 0.0
    
    return precision, recall


@step
def evaluate_model(
    svd_model: Any, knn_model: Any, baseline_model: Any, content_model: Dict[str, Any], raw_test_data: pd.DataFrame, k: int = 10
) -> Tuple[float, float, float, float, float, float, float, float, float, float, float, float, float, float]:
    test_data_tuples = [(d['userId'], d['id'], d['rating']) for d in raw_test_data.to_dict(orient='records')]

    svd_metrics = evaluate_model_predictions(svd_model, test_data_tuples, k)
    knn_metrics = evaluate_model_predictions(knn_model, test_data_tuples, k)
    baseline_metrics = evaluate_model_predictions(baseline_model, test_data_tuples, k)

    cosine_sim = content_model['cosine_sim']
    content_precision, content_recall = evaluate_content_based(raw_test_data, cosine_sim, k)
    print("SVD Metrics:", svd_metrics)
    print("KNN Metrics:", knn_metrics)
    print("Baseline Metrics:", baseline_metrics)
    print("Content-based Model Precision:", content_precision)
    print("Content-based Model Recall:", content_recall)
    return svd_metrics + knn_metrics + baseline_metrics + (content_precision, content_recall)