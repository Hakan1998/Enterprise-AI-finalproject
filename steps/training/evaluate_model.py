from typing import Any, Tuple, Dict, List
import pandas as pd
from collections import defaultdict
from surprise import accuracy
from zenml import step



"""
General Evaluation workflow:
1. Create rating predictions: For each user and each movie, generate predicted ratings using various recommendation models.
2. Evaluate predictions: Compare the predicted ratings with the actual ratings to compute evaluation metrics.
3. Calculate evaluation metrics: Use metrics such as RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), precision, and recall to measure the performance of the models.
4. Evaluate precision and recall: For each model, compute precision and recall at k, where k is the number of top recommendations considered. Precision measures the fraction of relevant items among the recommended items, and recall measures the fraction of relevant items that have been recommended out of all relevant items.
5. Evaluate content-based model: Additionally, evaluate a content-based model by computing cosine similarity between item features and using this similarity to make recommendations. Calculate precision and recall for the content-based model.
6. Output results: Return the evaluation metrics for all models, providing a comprehensive overview of their performance.
"""


def precision_recall_at_k(predictions: List[Tuple], k: int = 10, threshold: float = 3.5) -> Tuple[float, float]:
    """
    Calculate precision and recall at k for given predictions.

    Args:
        predictions: List of tuples containing user ID, item ID, true rating, estimated rating, and additional info.
        k: Number of top predictions to consider.
        threshold: Rating threshold to consider a recommendation relevant.

    Returns:
        Tuple containing precision and recall at k.
    """
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = min(k, len(user_ratings))
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:n_rec_k])
        
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    precision = sum(prec for prec in precisions.values()) / len(precisions)
    recall = sum(rec for rec in recalls.values()) / len(recalls)
    return precision, recall

def evaluate_model_predictions(model: Any, test_data: List[Tuple], k: int = 10) -> Tuple[float, float, float, float]:
    """
    Evaluate a model's predictions using RMSE, MAE, precision, and recall.

    Args:
        model: The recommendation model to evaluate.
        test_data: The test data as a list of tuples.
        k: Number of top predictions to consider for precision and recall.

    Returns:
        Tuple containing RMSE, MAE, precision, and recall.
    """
    predictions = model.test(test_data)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    precision, recall = precision_recall_at_k(predictions, k)
    return rmse, mae, precision, recall

def evaluate_content_based(raw_test_data: pd.DataFrame, cosine_sim: Any, k: int = 10) -> Tuple[float, float]:
    """
    Evaluate a content-based model's predictions using precision and recall.

    Args:
        raw_test_data: Raw test data in DataFrame format.
        cosine_sim: Cosine similarity matrix for content-based recommendations.
        k: Number of top recommendations to consider.

    Returns:
        Tuple containing precision and recall.
    """
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
                
                if actual_rating >= 2.9:
                    total_relevant += 1
                    if movie_id in recommended_movie_ids:
                        hits += 1
            except IndexError:
                continue
    
    precision = float(hits / total_recommended) if total_recommended > 0 else 0.0
    recall = float(hits / total_relevant) if total_relevant > 0 else 0.0
    
    return precision, recall

@step(enable_cache=False)
def evaluate_model(
    svd_model: Any, 
    knn_model: Any, 
    baseline_model: Any, 
    normal_model: Any, 
    nmf_model: Any, 
    slopeone_model: Any, 
    content_model: Dict[str, Any], 
    raw_test_data: pd.DataFrame, 
    k: int = 10
) -> Tuple[float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float]:
    """
    Evaluate multiple recommendation models and a content-based model.

    Args:
        svd_model: Trained SVD model.
        knn_model: Trained KNN model.
        baseline_model: Trained BaselineOnly model.
        normal_model: Trained NormalPredictor model.
        nmf_model: Trained NMF model.
        slopeone_model: Trained SlopeOne model.
        content_model: Trained content-based model.
        raw_test_data: Raw test data in DataFrame format.
        k: Number of top predictions to consider for precision and recall.

    Returns:
        Tuple containing evaluation metrics for all models.
    """
    test_data_tuples = [(d['userId'], d['id'], d['rating']) for d in raw_test_data.to_dict(orient='records')]

    svd_metrics = evaluate_model_predictions(svd_model, test_data_tuples, k)
    knn_metrics = evaluate_model_predictions(knn_model, test_data_tuples, k)
    baseline_metrics = evaluate_model_predictions(baseline_model, test_data_tuples, k)
    normal_metrics = evaluate_model_predictions(normal_model, test_data_tuples, k)
    nmf_metrics = evaluate_model_predictions(nmf_model, test_data_tuples, k)
    slopeone_metrics = evaluate_model_predictions(slopeone_model, test_data_tuples, k)

    cosine_sim = content_model['cosine_sim']
    content_precision, content_recall = evaluate_content_based(raw_test_data, cosine_sim, k)
    
    print("SVD Metrics:", svd_metrics)
    print("KNN Metrics:", knn_metrics)
    print("Baseline Metrics:", baseline_metrics)
    print("NormalPredictor Metrics:", normal_metrics)
    print("NMF Metrics:", nmf_metrics)
    print("SlopeOne Metrics:", slopeone_metrics)
    print("Content-based Model Precision:", content_precision)
    print("Content-based Model Recall:", content_recall)
    
    return svd_metrics + knn_metrics + baseline_metrics + normal_metrics + nmf_metrics + slopeone_metrics + (content_precision, content_recall)
