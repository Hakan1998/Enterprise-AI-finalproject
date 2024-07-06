from zenml import pipeline
from zenml.client import Client
from zenml.steps import step
import pandas as pd

from .feature_engineering_pipeline import (

    split_data,
    create_preprocessing_pipeline,
    feature_preprocessor
)


from steps.inference.load_inference_data import load_inference_data
from .training_pipeline import convert_to_surprise_format, hp_tuner, model_trainer
from steps.inference.get_recommendations import make_predictions
from steps.inference.load_best_model import load_best_model


@pipeline(enable_cache=False)
def inference_pipeline():

    dataset = load_inference_data()


    train_data, test_data = split_data(dataset)
    pipeline = create_preprocessing_pipeline(dataset)
    train_data, test_data, pipeline = feature_preprocessor(pipeline, train_data, test_data)

    # Die folgenden Zeilen sind dazu da, sicherzustellen, dass die korrekten Versionen der trainierten Daten verwendet werden
    client = Client()
    train_data = client.get_artifact_version("train_data_preprocessed")
    test_data = client.get_artifact_version("test_data_preprocessed")
    raw_train_data = train_data
    raw_test_data = test_data

    dataset, trainset, test_data = convert_to_surprise_format(raw_train_data=raw_train_data, raw_test_data=raw_test_data)

    # Load best model
    model = load_best_model()
    print(f"Model: {model}, Type: {type(model)}")
    print(f"Test Data: {test_data}, Type: {type(test_data)}")

    model = Client().get_artifact_version("best_model")
    """
    recommendations = make_predictions(
        model=model,
        test_data=test_data
    )
    
    print(recommendations)
    
    return recommendations

    """