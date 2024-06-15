from zenml import pipeline
from zenml.client import Client
from steps import prediction_service_loader, predictor, inference_data_loader, inference_preprocessing

@pipeline(enable_cache=False)
def inference_pipeline():
    """
    Runs the inference pipeline to predict the target variable of the inference data.
    """
    data = inference_data_loader("./data/inference.csv")
    client = Client()
    preprocessing_pipeline = client.get_artifact_version("pipeline")
    preprocessed_data = inference_preprocessing(preprocessing_pipeline, data)
    
    baseline_service = prediction_service_loader("baseline_model")
    baseline_predictions = predictor(baseline_service, preprocessed_data, "baseline")
    
    collaborative_service = prediction_service_loader("collaborative_model")
    collaborative_predictions = predictor(collaborative_service, preprocessed_data, "collaborative")
    
    content_based_service = prediction_service_loader("content_based_model")
    content_based_predictions = predictor(content_based_service, preprocessed_data, "content_based")
    
    matrix_factorization_service = prediction_service_loader("matrix_factorization_model")
    matrix_factorization_predictions = predictor(matrix_factorization_service, preprocessed_data, "matrix_factorization")

    print("Baseline Predictions:", baseline_predictions)
    print("Collaborative Predictions:", collaborative_predictions)
    print("Content-Based Predictions:", content_based_predictions)
    print("Matrix Factorization Predictions:", matrix_factorization_predictions)