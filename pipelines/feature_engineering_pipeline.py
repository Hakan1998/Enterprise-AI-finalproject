from steps import load_movie_data,clean_movie_data,load_rating_data,preprocess_rating_data,merged_data,split_data,create_preprocessing_pipeline,feature_preprocessor
from zenml import pipeline
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

"""
This pipeline will perform feature engineering on our movie and rating dataset. we combine the two dataset only with valid movieID

Take a look at the create_preprocessing_pipeline step and the feature_preprocessor step. 
You will need to fix some parts there. 
"""
@pipeline(enable_cache=True)
def feature_engineering_pipeline():
    """"
        Pipeline function for performing feature engineering combined data of movie data
        and user data.
    """
    raw_movies = load_movie_data("./data/movies_metadata.csv")
    movies = clean_movie_data(raw_movies)
    raw_users = load_rating_data("./data/ratings_small.csv")
    users = preprocess_rating_data(raw_users)

    dataset = merged_data(movies,users)

    train_data,test_data = split_data(dataset)
    pipeline = create_preprocessing_pipeline(dataset)
    train_data,test_data,pipeline = feature_preprocessor(pipeline,train_data,test_data)

#EMAIL Alert to keep the team informed about the process     
# Funktion zum Senden der Email
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

# E-Mail Konfiguration
sender_email = "Enterprise_AI@gmx.de"
receiver_email = "fink.silas@gmx.de"
password = "EnterpriseAI_Gruppe4"

# Sende eine E-Mail nach erfolgreicher Pipeline-Ausführung
subject = "Feature Engineering Pipeline Successful"
body = "The feature engineering pipeline has been successfully executed."
send_email(sender_email, receiver_email, password, subject, body)

# Ausführung der Pipeline
if __name__ == "__main__":
    feature_engineering_pipeline()
