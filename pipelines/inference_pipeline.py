from zenml import pipeline
from zenml.client import Client
from zenml.steps import step
import pandas as pd
from steps.inference.load_best_model import load_best_model
from steps.inference.load_and_preprocess_inference_data import load_and_preprocess_inference_data
from steps.inference.make_predictions import make_predictions
from steps.training.convert_to_surprise_format import convert_to_surprise_format
from steps.inference.get_test_data import get_test_data_series

@pipeline(enable_cache=False)
def inference_pipeline():



    # Load and preprocess the new inference data all in one step
    preprocess_inference_data = load_and_preprocess_inference_data() 

    # Get the best model from model evaluation
    best_model = load_best_model()

    # prediction function doesnt work properly -> TypeError: issubclass() arg 1 must be a class
    #recommendations = make_predictions(model=best_model, raw_test_data=preprocess_inference_data)

    #return recommendations

"""
    data and model would be ready here

   #  recommendations = make_predictions(model=best_model, raw_test_data=preprocess_inference_data)

   # TypeError: issubclass() arg 1 must be a class
   # --> couldnt fix this tried different custom materializer, setting up other steps for solving etc. nothing worked
   # --> even trying many different things it seems not really possible to set up the saved model from deployment since surprise and zenml have different data types
   # --> so experiment tracking / service loader etc. wouldnt make sense here


"""

