from zenml import step
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import joblib
from typing import Any

# Baseline Model
@step
def train_baseline_model(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    mean_rating = y_train.mean()
    joblib.dump(mean_rating, "baseline_model.joblib")
    return mean_rating

# Kollaboratives Filtern
@step
def train_collaborative_model(X_train: pd.DataFrame, y_train: pd.Series) -> NearestNeighbors:
    user_item_matrix = X_train.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(user_item_matrix)
    joblib.dump(model, "collaborative_model.joblib")
    return model

# Content-Based Filtern
@step
def train_content_based_model(X_train: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
    similarity_matrix = cosine_similarity(X_train, X_train)
    joblib.dump(similarity_matrix, "content_based_model.joblib")
    return similarity_matrix

# Matrix Faktorisierung
@step
def train_matrix_factorization_model(X_train: pd.DataFrame, y_train: pd.Series) -> NMF:
    user_item_matrix = X_train.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    model = NMF(n_components=20, init='random', random_state=42)
    model.fit(user_item_matrix)
    joblib.dump(model, "matrix_factorization_model.joblib")
    return model