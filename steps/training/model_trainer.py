from zenml import step
from surprise import KNNBasic, SVD, Dataset, Reader
from surprise.model_selection import train_test_split as surprise_train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple

@step
def model_trainer(
    train_data: pd.DataFrame, 
    test_data: pd.DataFrame, 
    best_params: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]]]:
    
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train_data[['userId', 'movieId', 'rating']], reader)
    trainset, _ = surprise_train_test_split(data, test_size=0.2)

    # User CF
    user_cf = KNNBasic(**best_params['user_cf'])
    user_cf.fit(trainset)
    
    def get_user_cf_recommendations(user_id: int, n: int = 10) -> List[int]:
        user_rated_movies = test_data[test_data.userId == user_id].movieId.values
        all_movies = set(train_data.movieId.unique())
        unrated_movies = list(all_movies - set(user_rated_movies))

        predictions = [user_cf.predict(user_id, movie_id).est for movie_id in unrated_movies]
        top_n_indices = np.argsort(predictions)[-n:]
        top_n_movies = [unrated_movies[i] for i in top_n_indices]

        return top_n_movies

    user_cf_recs = {user_id: get_user_cf_recommendations(user_id) for user_id in test_data.userId.unique()}

    # Item CF
    item_cf = KNNBasic(**best_params['item_cf'])
    item_cf.fit(trainset)
    
    def get_item_cf_recommendations(user_id: int, n: int = 10) -> List[int]:
        user_rated_movies = test_data[test_data.userId == user_id].movieId.values
        all_movies = set(train_data.movieId.unique())
        unrated_movies = list(all_movies - set(user_rated_movies))

        predictions = [item_cf.predict(user_id, movie_id).est for movie_id in unrated_movies]
        top_n_indices = np.argsort(predictions)[-n:]
        top_n_movies = [unrated_movies[i] for i in top_n_indices]

        return top_n_movies

    item_cf_recs = {user_id: get_item_cf_recommendations(user_id) for user_id in test_data.userId.unique()}

    # SVD
    svd = SVD(**best_params['svd'])
    svd.fit(trainset)
    
    def get_svd_recommendations(user_id: int, n: int = 10) -> List[int]:
        user_rated_movies = test_data[test_data.userId == user_id].movieId.values
        all_movies = set(train_data.movieId.unique())
        unrated_movies = list(all_movies - set(user_rated_movies))

        predictions = [svd.predict(user_id, movie_id).est for movie_id in unrated_movies]
        top_n_indices = np.argsort(predictions)[-n:]
        top_n_movies = [unrated_movies[i] for i in top_n_indices]

        return top_n_movies

    svd_recs = {user_id: get_svd_recommendations(user_id) for user_id in test_data.userId.unique()}

    # Content-Based
    train_data['combined_features'] = train_data['genres'] + ' ' + train_data['director'] + ' ' + train_data['actors'] + ' ' + train_data['title']
    tfidf = TfidfVectorizer(stop_words=best_params['content']['stop_words'], ngram_range=best_params['content']['ngram_range'])
    tfidf_matrix = tfidf.fit_transform(train_data['combined_features'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    def get_content_based_recommendations(user_id: int, n: int = 10) -> List[int]:
        user_rated_movies = test_data[test_data.userId == user_id].movieId.values
        if len(user_rated_movies) == 0:
            return []  # No recommendations if the user has not rated any movies

        rated_indices = [train_data[train_data.movieId == movie_id].index[0] for movie_id in user_rated_movies]
        sim_scores = cosine_sim[rated_indices].mean(axis=0)
        unrated_indices = [i for i in range(len(sim_scores)) if train_data.iloc[i].movieId not in user_rated_movies]
        sim_scores = [(i, sim_scores[i]) for i in unrated_indices]

        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_n_indices = [i[0] for i in sim_scores[:n]]
        top_n_movies = train_data.iloc[top_n_indices].movieId.tolist()

        return top_n_movies

    content_recs = {user_id: get_content_based_recommendations(user_id) for user_id in test_data.userId.unique()}

    return user_cf_recs, item_cf_recs, svd_recs, content_recs
