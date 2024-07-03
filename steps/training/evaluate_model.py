from typing import Any, Tuple, Dict, List
import pandas as pd
from collections import defaultdict
from surprise import accuracy
from zenml import step

def precision_recall_at_k(predictions: List[Tuple], k: int = 10, threshold: float = 3.5) -> Tuple[float, float]:
    """
    Calculate precision and recall at k for given predictions.

    Args:
        predictions (List[Tuple]): List of predictions in the format (user_id, item_id, true_rating, est_rating, details).
        k (int): Number of top recommendations to consider.
        threshold (float): Threshold above which a rating is considered positive.

    Returns:
        Tuple[float, float]: Precision and recall at k.
    """
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

def evaluate_model_predictions(model: Any, test_data: List[Tuple], k: int = 10) -> Tuple[float, float, float, float]:
    """
    Evaluate a given model using test data and calculate RMSE, MAE, precision, and recall.

    Args:
        model (Any): The model to evaluate.
        test_data (List[Tuple]): Test data in the format (user_id, item_id, true_rating).
        k (int): Number of top recommendations to consider.

    Returns:
        Tuple[float, float, float, float]: RMSE, MAE, precision, and recall for the model.
    """
    predictions = model.test(test_data)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    precision, recall = precision_recall_at_k(predictions, k)
    return rmse, mae, precision, recall

def evaluate_content_based(raw_test_data: pd.DataFrame, cosine_sim: Any, k: int = 10) -> Tuple[float, float]:
    """
    Evaluate content-based filtering using cosine similarity.

    Args:
        raw_test_data (pd.DataFrame): The raw test data.
        cosine_sim (Any): Cosine similarity matrix.
        k (int): Number of top recommendations to consider.

    Returns:
        Tuple[float, float]: Precision and recall for content-based filtering.
    """
    hits = 0
    total = 0
    
    for _, row in raw_test_data.iterrows():
        movie_id = row['movieId']
        actual_rating = row['rating']
        
        if movie_id not in raw_test_data['movieId'].values:
            continue

        # Validate index
        try:
            idx = raw_test_data.index[raw_test_data['movieId'] == movie_id][0]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:k+1]

            # Compute precision and recall
            recommended_movie_ids = [raw_test_data.iloc[i[0]]['movieId'] for i in sim_scores]
            if actual_rating >= 4.0:  # Assuming ratings of 4 and above are positive
                total += 1
                if movie_id in recommended_movie_ids:
                    hits += 1
        except IndexError:
            continue
    
    precision = float(hits / total) if total > 0 else 0.0
    recall = float(hits / total) if total > 0 else 0.0
    
    return precision, recall

@step
def evaluate_model(
    svd_model: Any, knn_model: Any, baseline_model: Any, content_model: Dict[str, Any], raw_test_data: pd.DataFrame
) -> Tuple[float, float, float, float, float, float, float, float, float, float, float, float, float, float]:
    """
    Evaluate multiple models (SVD, KNN, Baseline, Content-based) and return their performance metrics.

    Args:
        svd_model (Any): SVD model to evaluate.
        knn_model (Any): KNN model to evaluate.
        baseline_model (Any): Baseline model to evaluate.
        content_model (Dict[str, Any]): Content-based model containing the cosine similarity matrix.
        raw_test_data (pd.DataFrame): Raw test data.

    Returns:
        Tuple[float, float, float, float, float, float, float, float, float, float, float, float, float, float]: 
        Performance metrics (RMSE, MAE, precision, recall) for all models.
    """
    # Convert test_data to list of tuples
    test_data_tuples = [(d['userId'], d['movieId'], d['rating']) for d in raw_test_data.to_dict(orient='records')]

    # Evaluate SVD model
    svd_metrics = evaluate_model_predictions(svd_model, test_data_tuples)

    # Evaluate KNN model
    knn_metrics = evaluate_model_predictions(knn_model, test_data_tuples)

    # Evaluate Baseline model
    baseline_metrics = evaluate_model_predictions(baseline_model, test_data_tuples)

    # Unpack the content model
    cosine_sim = content_model['cosine_sim']
    
    # Evaluate content-based filtering
    content_precision, content_recall = evaluate_content_based(raw_test_data, cosine_sim, k=10)

    return svd_metrics + knn_metrics + baseline_metrics + (content_precision, content_recall)