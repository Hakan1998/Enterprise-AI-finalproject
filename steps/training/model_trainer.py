import pandas as pd
import re
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse import hstack
from surprise import SVD, KNNBasic, BaselineOnly, NormalPredictor, NMF, SlopeOne, Trainset
from typing import Dict, Any, Tuple, Annotated
from zenml import step
import mlflow

# Ensure nltk data is downloaded
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    """
    Preprocess the input text by converting to lowercase, removing HTML tags, punctuation, numbers, stopwords, and lemmatizing the words.
    Content Based filtering specific so its in the basic FE Pipeline.

    Args:
        text: The input text to preprocess.

    Returns:
        The cleaned and preprocessed text.
    """
    # Convert to lowercase, # Remove HTML tags, # Remove HTML tags, # Remove stopwords and lemmatize

    text = text.lower()
    text = BeautifulSoup(text, "html.parser").get_text()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\d+', '', text)
    
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    # Join the cleaned words back into a single string
    cleaned_text = " ".join(cleaned_words)
    
    return cleaned_text

@step(enable_cache=False)
def model_trainer(
    train_data: Trainset, 
    raw_train_data: pd.DataFrame, 
    best_params_svd: Dict[str, Any], 
    best_params_knn: Dict[str, Any], 
    best_params_baseline: Dict[str, Any], 
    best_params_normal: Dict[str, Any],
    best_params_nmf: Dict[str, Any],
    best_params_slope_one: Dict[str, Any],
    content_model_params: Dict[str, Any]
) -> Tuple[
    Annotated[SVD, "SVD Model"],
    Annotated[KNNBasic, "KNN Model"],
    Annotated[BaselineOnly, "Baseline Only Model"],
    Annotated[NormalPredictor, "Normal Predictor Model"],
    Annotated[NMF, "NMF Model"],
    Annotated[SlopeOne, "Slope One Model"],
    Annotated[Any, "Content-based Model"]
]:
    """
    Train various recommendation models from the Surprise Library and a content-based model using the provided hyperparameters.

    Args:
        train_data: The training data in the format expected by the surprise library.
        raw_train_data: Raw training data in DataFrame format.
        best_params_svd: Best hyperparameters for the SVD model.
        best_params_knn: Best hyperparameters for the KNN model.
        best_params_baseline: Best hyperparameters for the BaselineOnly model.
        best_params_normal: Best hyperparameters for the NormalPredictor model.
        best_params_nmf: Best hyperparameters for the NMF model.
        best_params_slope_one: Best hyperparameters for the SlopeOne model.
        content_model_params: Parameters for the content-based model.

    Returns:
        A tuple containing the trained SVD, KNN, BaselineOnly, NormalPredictor, NMF, SlopeOne, and content-based models.
    """
    mlflow.sklearn.autolog() # This will auto-log the model training process and save it to MLflow 
    mlflow.log_artifact
    mlflow.log_metric
    mlflow.log_param

    
    # Train the collaborative filtering models
    svd = SVD(**best_params_svd)
    knn = KNNBasic(**best_params_knn)
    baseline = BaselineOnly(**best_params_baseline)
    normal_predictor = NormalPredictor(**best_params_normal)
    nmf = NMF(**best_params_nmf)
    slope_one = SlopeOne(**best_params_slope_one)

    svd.fit(train_data)
    knn.fit(train_data)
    baseline.fit(train_data)
    normal_predictor.fit(train_data)
    nmf.fit(train_data)
    slope_one.fit(train_data)

    # Ensure ngram_range is a tuple for content model parameters
    content_model_params['ngram_range'] = tuple(content_model_params['ngram_range'])

    # Combine 'title', 'tagline', and 'overview' into one text column for content-based filtering
    raw_train_data['combined_text'] = (
        raw_train_data['title'].fillna('') + ' ' +
        raw_train_data['tagline'].fillna('') + ' ' +
        raw_train_data['overview'].fillna('')
    )

    # Create TF-IDF matrix for 'combined_text'
    tfidf = TfidfVectorizer(stop_words='english', **content_model_params)
    tfidf_matrix = tfidf.fit_transform(raw_train_data['combined_text'])

    # if we want to also use numerical columns in content-based model, but it's mostly uncommon
    # numerical_features = raw_train_data[['budget', 'revenue', 'runtime']].fillna(0).values

    # Combine TF-IDF matrix with numerical features if used
    combined_features = hstack([tfidf_matrix])

    # Compute cosine similarity on the combined features
    cosine_sim = linear_kernel(combined_features, combined_features)

    content_model = {
        'combined_features': combined_features,
        'cosine_sim': cosine_sim
    }

    return svd, knn, baseline, normal_predictor, nmf, slope_one, content_model
