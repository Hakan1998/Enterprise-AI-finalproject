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
    split_data,
    create_preprocessing_pipeline,
    feature_preprocessor
)

from typing import Tuple, Any
"""
Load and preprocess the Inference data. Identicly as Feature Engineering pipeline- 
"""
@step
def load_and_preprocess_inference_data() -> Annotated[pd.DataFrame, "inference_data"]:

    raw_movies = load_movie_data("./data/movies_metadata.csv")
    movies = clean_movie_data(raw_movies)
    raw_users = load_rating_data("./data/inference_ratings.csv") # here now new inference data
    users = preprocess_rating_data(raw_users)

    dataset = merged_data(movies, users)

    train_data,test_data = split_data(dataset)
    pipeline = create_preprocessing_pipeline(dataset)
    train_data,test_data,pipeline = feature_preprocessor(pipeline,train_data,test_data)
    combined_data = pd.concat([train_data, test_data], ignore_index=True)
    print(combined_data)                                        # no need to split anymore but its easier if we run our previous functions and then just merge the data, Reason is zenml sets many artifacts so its difficult to edit things after.


    

    return combined_data