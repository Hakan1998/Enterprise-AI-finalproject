import pandas as pd
import re
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse import hstack
from surprise import SVD, KNNBasic, BaselineOnly, Trainset
from typing import Dict, Any, Tuple
from zenml import step

# Ensure nltk data is downloaded
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Remove numbers
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
    content_model_params: Dict[str, Any]
) -> Tuple[SVD, KNNBasic, BaselineOnly, Dict[str, Any]]:
    
    # Train collaborative filtering models
    svd = SVD(**best_params_svd)
    knn = KNNBasic(**best_params_knn)
    baseline = BaselineOnly(**best_params_baseline)

    svd.fit(train_data)
    knn.fit(train_data)
    baseline.fit(train_data)

    # Ensure ngram_range is a tuple
    content_model_params['ngram_range'] = tuple(content_model_params['ngram_range'])

    # Combine 'title', 'tagline', and 'overview' into one text column
    raw_train_data['combined_text'] = (
        raw_train_data['title'].fillna('') + ' ' +
        raw_train_data['tagline'].fillna('') + ' ' +
        raw_train_data['overview'].fillna('')
    )

    # Preprocess the combined text
    raw_train_data['cleaned_text'] = raw_train_data['combined_text'].apply(preprocess_text)

    # Create TF-IDF matrix for 'cleaned_text'
    tfidf = TfidfVectorizer(stop_words='english', **content_model_params)
    tfidf_matrix = tfidf.fit_transform(raw_train_data['cleaned_text'])

    # Compute cosine similarity on the TF-IDF matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    content_model = {
        'tfidf_matrix': tfidf_matrix,
        'cosine_sim': cosine_sim
    }

    return svd, knn, baseline, content_model
