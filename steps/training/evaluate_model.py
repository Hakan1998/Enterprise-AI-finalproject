from zenml import step
from typing import Dict, Any
import pandas as pd

@step
def evaluate_model(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    user_cf_recs: Dict[int, Any],
    item_cf_recs: Dict[int, Any],
    svd_recs: Dict[int, Any],
    content_recs: Dict[int, Any]
) -> Dict[str, Any]:
    


    def calculate_ndcg(recommended_items, true_items, k):
        dcg = 0.0
        for i, item in enumerate(recommended_items):
            if item in true_items:
                dcg += 1 / np.log2(i + 2)  # i+2 because of 0-based index and log base 2
        idcg = sum([1 / np.log2(i + 2) for i in range(min(len(true_items), k))])
        return dcg / idcg if idcg > 0 else 0

    def precision_recall_f1_ndcg_at_k(model_func, k=10):
        hits = 0
        total_relevant = 0
        total_recommended = 0
        ndcg = 0.0

        user_groups = test_data.groupby('userId')

        for user_id, group in user_groups:
            true_items = set(group.movieId.values)
            recommended_items = set(model_func(user_id, k))

            hits += len(true_items & recommended_items)
            total_relevant += len(true_items)
            total_recommended += len(recommended_items)

            ndcg += calculate_ndcg(recommended_items, true_items, k)

        precision = hits / total_recommended if total_recommended > 0 else 0
        recall = hits / total_relevant if total_relevant > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
        avg_ndcg = ndcg / len(user_groups)

        return precision, recall, f1, avg_ndcg

    results = {}
    results['user_cf'] = precision_recall_f1_ndcg_at_k(lambda user_id, n: user_cf_recs[user_id], k=10)
    results['item_cf'] = precision_recall_f1_ndcg_at_k(lambda user_id, n: item_cf_recs[user_id], k=10)
    results['svd'] = precision_recall_f1_ndcg_at_k(lambda user_id, n: svd_recs[user_id], k=10)
    results['content_based'] = precision_recall_f1_ndcg_at_k(lambda user_id, n: content_recs[user_id], k=10)
    results['hybrid'] = precision_recall_f1_ndcg_at_k(lambda user_id, n: user_cf_recs[user_id] + item_cf_recs[user_id] + svd_recs[user_id] + content_recs[user_id], k=10)

    return results
