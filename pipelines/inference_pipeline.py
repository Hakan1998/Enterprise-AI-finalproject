from zenml import pipeline
from zenml.client import Client
from zenml import steps
import pandas as pd

from .feature_engineering_pipeline import (
    load_movie_data,
    clean_movie_data,
    load_rating_data,
    preprocess_rating_data,
    merged_data,
    split_data,
    create_preprocessing_pipeline,
    feature_preprocessor
)

from .training_pipeline import convert_to_surprise_format, evaluate_model,hp_tuner, model_trainer
from steps.inference.get_recommendations import get_model_recommendations
def inference_pipeline():

    def diagnose_data_types(data: pd.DataFrame):
        print(dataset.dtypes)


    raw_movies = load_movie_data("./data/movies_metadata.csv")
    movies = clean_movie_data(raw_movies)
    raw_users = load_rating_data("./data/inference_ratings.csv")
    users = preprocess_rating_data(raw_users)

    dataset = merged_data(movies,users)

    # Diagnose der Datentypen vor dem Preprocessing
    diagnose_data_types(dataset)

    train_data,test_data = split_data(dataset)
    pipeline = create_preprocessing_pipeline(dataset)
    train_data,test_data,pipeline = feature_preprocessor(pipeline,train_data,test_data)

    client = Client()
    train_data = client.get_artifact_version("train_data_preprocessed")
    test_data = client.get_artifact_version("test_data_preprocessed")
    raw_train_data = train_data
    raw_test_data = test_data

    dataset, trainset, test_data = convert_to_surprise_format(raw_train_data=raw_train_data, raw_test_data=raw_test_data)
    best_params_svd, best_params_knn, best_params_baseline, content_model_params = hp_tuner(dataset=dataset, raw_train_data=raw_train_data)
    svd_model, knn_model, baseline_model, content_model = model_trainer(
        train_data=trainset, 
        raw_train_data=raw_train_data,
        best_params_svd=best_params_svd, 
        best_params_knn=best_params_knn, 
        best_params_baseline=best_params_baseline,
        content_model_params=content_model_params
    )

    recommendations = get_recommendations(
        svd_model=svd_model, 
        knn_model=knn_model, 
        baseline_model=baseline_model, 
        content_model=content_model, 
        raw_test_data=raw_test_data
    )

    return recommendations