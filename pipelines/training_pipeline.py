from zenml import pipeline
from steps.training.hp_tuner import hp_tuner
from steps.training.model_trainer import model_trainer
from steps.training.evaluate_model import evaluate_model
from steps.training.convert_to_surprise_format import convert_to_surprise_format
from zenml.client import Client
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

@pipeline(enable_cache=True)
def training_pipeline():
    """
    Training pipeline for building and evaluating recommendation models using the Surprise library.

    This pipeline includes the following steps:
    
    1. Load preprocessed training and testing data artifacts.
    2. Convert the raw data into the format required by the Surprise library.
    3. Perform hyperparameter tuning to find the best parameters for different algorithms.
    4. Train multiple recommendation models using the best hyperparameters.
    5. Evaluate the trained models on the test data and log the results.

    """
    client = Client()
    train_data = client.get_artifact_version("train_data_preprocessed")
    test_data = client.get_artifact_version("test_data_preprocessed")
    raw_train_data = train_data
    raw_test_data = test_data

    dataset, trainset, test_data = convert_to_surprise_format(raw_train_data=raw_train_data, raw_test_data=raw_test_data)
    best_params_svd, best_params_knn, best_params_baseline, best_params_normal, best_params_nmf, best_params_slope_one, content_model_params = hp_tuner(dataset=dataset, raw_train_data=raw_train_data)
    
    svd_model, knn_model, baseline_model, normal_model, nmf_model, slopeone_model, content_model = model_trainer(
        train_data=trainset, 
        raw_train_data=raw_train_data,
        best_params_svd=best_params_svd, 
        best_params_knn=best_params_knn, 
        best_params_baseline=best_params_baseline,
        best_params_normal=best_params_normal,
        best_params_nmf=best_params_nmf,
        best_params_slope_one=best_params_slope_one,
        content_model_params=content_model_params
    )
    
    evaluate_model(
        svd_model=svd_model, 
        knn_model=knn_model, 
        baseline_model=baseline_model, 
        normal_model=normal_model,
        nmf_model=nmf_model,
        slopeone_model=slopeone_model,
        content_model=content_model, 
        raw_test_data=raw_test_data
    )

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
subject = "Training Pipeline"
body = "The training pipeline has been executed."
send_email(sender_email, receiver_email, password, subject, body)

# Execution of the pipeline
if __name__ == "__main__":
    training_pipeline()
