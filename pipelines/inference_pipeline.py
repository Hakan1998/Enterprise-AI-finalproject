from zenml import pipeline
from zenml.client import Client
from zenml.steps import step
import pandas as pd
from steps.inference.load_best_model import load_best_model
from steps.inference.load_and_preprocess_inference_data import load_and_preprocess_inference_data
from steps.inference.make_recommendations import make_recommendations
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from steps.utility import SendEmailStep




"""
Inference Pipeline that predict the top K movie matches for each User one the inference data
"""

@pipeline(enable_cache=False)
def inference_pipeline():


    # Load and preprocess the new inference data all in one step. Since it is the same as in Feature Engineering theres no need to create multiple steps
    preprocess_inference_data = load_and_preprocess_inference_data() 


    """
    Hint for Loading an Surprise Model zenml Artifacts : 

    The original load_best_model function failed because it tried to use mlflow.surprise.load_model, which doesn't exist in MLflow, and had issues handling model files.
    We fixed it by changing the function to download the whole model folder with mlflow.artifacts.download_artifacts. 

    So we had to create an custom MLflow Python model wrapper for the Surprise skikit Library. 
    
    This ensures the model loads correctly, even if we use the latest version instead of the production version.
    --> for more details look into the load_best_model funtion
    """
   
    best_model = load_best_model()

    # make the top k movie recommendations for each User
    top_k_recommendations = make_recommendations(model=best_model, raw_test_data=preprocess_inference_data, k=10)

    #Define email subject and body for this specific pipeline
    subject = "Inference Pipeline Successful"
    body = "The Inference pipeline has been successfully executed."

    send_email_step = SendEmailStep(subject=subject, body=body)
    send_email_step.entrypoint()



    return top_k_recommendations


    