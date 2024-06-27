from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from zenml import step

class DictionaryExtractor(TransformerMixin):
    """Extracts names from a list of dictionaries within DataFrame columns and binarizes them."""
    def __init__(self, key='name'):
        self.key = key
        self.mlb = MultiLabelBinarizer()
    
    def fit(self, X, y=None):
        # Aggregate all names from the dictionaries across all rows to fit the MultiLabelBinarizer
        names = [item[self.key] for sublist in X for item in sublist if self.key in item]
        self.mlb.fit([names])
        return self

    def transform(self, X):
        # Transform each list of dictionaries to a list of names
        transformed = X.apply(lambda sublist: [item[self.key] for item in sublist if self.key in item])
        return self.mlb.transform(transformed)


@step
def create_preprocessing_pipeline(dataset: pd.DataFrame) -> Pipeline:
    # Numeric features preprocessing
    numeric_features = dataset[['popularity', 'revenue', 'runtime', 'vote_average', 'vote_count',"user_average_rating","user_rating_count","rating","weighted_rating","user_average_rating","user_rating_count"]]
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical features preprocessing
    categorical_features = dataset[['original_language', 'status']]
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ("encoder",OneHotEncoder(sparse_output=False,handle_unknown="ignore"))
    ])

    # Complex encoded features
    complex_features = ['genres', 'production_companies', 'production_countries']
    complex_transformer = Pipeline([
    ('extractor_and_binarizer', DictionaryExtractor()),  # Handles extraction and binarization in one step
    ])
    
    # Text features preprocessing
    text_features = dataset[['title', 'overview']]
    text_transformer = Pipeline(steps=[
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('svd', TruncatedSVD(n_components=100))  # Dimensionality reduction
    ])
    
    # Combine all transformers
    preprocessing = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('complex', complex_transformer, complex_features),
            ('text', text_transformer, text_features)
        ])
    pipeline = Pipeline([
        ("preprocessing",preprocessing)
    ])

    return pipeline

