import pandas as pd
from surprise import SVD, KNNBasic, BaselineOnly
from typing import Any, Dict, List
from zenml import step

def predict_ratings(model: Any, data: pd.DataFrame) -> List[Dict[str, Any]]:
    predictions = []
    for _, row in data.iterrows():
        user_id = row['userId']
        movie_id = row['id']
        pred = model.predict(uid=user_id, iid=movie_id)
        predictions.append({'userId': user_id, 'movieId': movie_id, 'predicted_rating': pred.est})
    return predictions

def predict_content_based(user_id: int, cosine_sim: Any, raw_data: pd.DataFrame, k: int = 10) -> List[Dict[str, Any]]:
    user_data = raw_data[raw_data['userId'] == user_id]
    predictions = []

    for _, row in user_data.iterrows():
        movie_id = row['id']
        idx = raw_data.index[raw_data['id'] == movie_id][0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:k+1]

        for i in sim_scores:
            sim_movie_id = raw_data.iloc[i[0]]['id']
            predictions.append({'userId': user_id, 'movieId': sim_movie_id, 'similarity': i[1]})
    return predictions

@step(enable_cache=False)
def make_predictions(
    svd_model: SVD, knn_model: KNNBasic, baseline_model: BaselineOnly, content_model: Dict[str, Any], inference_data: pd.DataFrame
) -> pd.DataFrame:
    svd_predictions = predict_ratings(svd_model, inference_data)
    knn_predictions = predict_ratings(knn_model, inference_data)
    baseline_predictions = predict_ratings(baseline_model, inference_data)

    user_ids = inference_data['userId'].unique()
    content_predictions = []
    for user_id in user_ids:
        content_predictions.extend(predict_content_based(user_id, content_model['cosine_sim'], inference_data))

    predictions = {
        'svd_predictions': svd_predictions,
        'knn_predictions': knn_predictions,
        'baseline_predictions': baseline_predictions,
        'content_predictions': content_predictions
    }

    return pd.DataFrame(predictions)
