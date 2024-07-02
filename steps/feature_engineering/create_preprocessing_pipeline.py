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



@step
def create_preprocessing_pipeline(dataset: pd.DataFrame) -> Pipeline:
    # Numeric features preprocessing
    #numeric_features = ["budget",'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count',"user_average_rating","user_rating_count","rating","weighted_rating"]
    
    numerical_features = dataset.select_dtypes(exclude=['object']).columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical features preprocessing
    #categorical_features = ['original_language', 'status',"id","userId"]
    categorical_features = dataset.select_dtypes(include=['object']).columns
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ("encoder",OneHotEncoder(sparse_output=False,handle_unknown="ignore"))
    ])

    # Complex encoded features
    #complex_features = ['genres']
    #complex_transformer = Pipeline([
    #('extractor_and_binarizer', DictionaryExtractor()),  # Handles extraction and binarization in one step
    #])
    
    # Text features preprocessing
    #text_features = ['title', 'overview']
    #text_transformer = Pipeline(steps=[
        #('tfidf', TfidfVectorizer(stop_words='english')),
        #('svd', TruncatedSVD(n_components=2))  # Dimensionality reduction
    #])
    
    # Combine all transformers
    preprocessing = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])
            #('complex', complex_transformer, complex_features),
            #('text', text_transformer, text_features)
        
    pipeline = Pipeline([
        ("preprocessing",preprocessing)
    ])

    return pipeline

