from zenml import step
from typing_extensions import Annotated
from zenml.client import Client
import pandas as pd


from pipelines.feature_engineering_pipeline import (
    load_movie_data,
    clean_movie_data,
    load_rating_data,
    preprocess_rating_data,
    merged_data,
)

from typing import Tuple, Any

@step
def load_inference_data() -> Annotated[pd.DataFrame, "inference_data"]:
    raw_movies = load_movie_data("./data/movies_metadata.csv")
    movies = clean_movie_data(raw_movies)
    raw_users = load_rating_data("./data/inference_ratings.csv")
    users = preprocess_rating_data(raw_users)

    dataset = merged_data(movies, users)
    return dataset