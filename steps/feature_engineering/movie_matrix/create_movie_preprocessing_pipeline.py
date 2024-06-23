from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from zenml.steps import step

# Custom transformers if necessary
class MultiLabelBinarizerWrapper:
    def fit(self, X, y=None):
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(X)
        return self

    def transform(self, X):
        return self.mlb.transform(X)

@step
def create_preprocessing_pipeline():
    # Numeric features preprocessing
    numeric_features = ['popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical features preprocessing
    categorical_features = ['original_language', 'status']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ("encoder",OneHotEncoder(sparse_output=False,handle_unknown="ignore"))
    ])

    # Complex encoded features
    complex_features = ['genres', 'production_companies', 'production_countries']
    complex_transformer = Pipeline(steps=[
        ('extractor', MultiLabelBinarizerWrapper()),  # Assuming proper extraction function is implemented
    ])
    
    # Text features preprocessing
    text_features = ['title', 'overview']
    text_transformer = Pipeline(steps=[
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('svd', TruncatedSVD(n_components=1000))  # Dimensionality reduction
    ])
    
    # Combine all transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('complex', complex_transformer, complex_features),
            ('text', text_transformer, text_features)
        ])

    return preprocessor

