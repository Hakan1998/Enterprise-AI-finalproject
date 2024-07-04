from zenml import pipeline
from zenml.client import Client
from steps.inference.load_inference_rating import load_inference_rating
from steps.feature_engineering.feature_preprocessor import feature_preprocessor
#from steps.training.model_trainer import load_trained_models
from steps.inference.make_predictions import make_predictions

@pipeline(enable_cache=False)
def inference_pipeline():
    client = Client()

    # Load the preprocessed data for inference
    raw_inference_data = load_inference_rating("./data/inference_ratings.csv")
    
    # Load the feature preprocessing pipeline from previous steps
    preprocessing_pipeline = client.get_pipeline("feature_engineering_pipeline")

    # Preprocess the inference data
    _, inference_data, _ = feature_preprocessor(preprocessing_pipeline, None, raw_inference_data)

    # Load trained models
    svd_model, knn_model, baseline_model, content_model = load_trained_models()

    # Make predictions on the inference data
    predictions = make_predictions(
        svd_model=svd_model, 
        knn_model=knn_model, 
        baseline_model=baseline_model, 
        content_model=content_model, 
        inference_data=inference_data
    )

    return predictions
