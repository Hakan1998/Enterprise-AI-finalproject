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


    return top_k_recommendations

#EMAIL Alert to keep the team informed about the process    
# Function for sending emails
def send_email(sender_email, receiver_email, password, subject, body):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('mail.gmx.de', 587)
        server.starttls()
        server.login(sender_email, password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        print("Email erfolgreich gesendet!")
    except Exception as e:
        print(f"Fehler beim Senden der Email: {e}")

# E-Mail konfiguration
sender_email = "Enterprise_AI@gmx.de"
receiver_email = "fink.silas@gmx.de"
password = "EnterpriseAI_Gruppe4"

# Send an email upon successful pipeline execution
subject = "Inference Pipeline"
body = "The inference pipeline has been executed."
send_email(sender_email, receiver_email, password, subject, body)

# Execution of the pipeline
if __name__ == "__main__":
    inference_pipeline()
