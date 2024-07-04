from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
from zenml import step
from typing_extensions import Annotated
from typing import Tuple


@step(enable_cache=False)
def create_preprocessing_pipeline(dataset: pd.DataFrame) -> Pipeline:
    # Columns that should not be transformed
    passthrough_columns = ['rating', 'id', 'userId', 'overview', "title", "tagline"]
    
    # Identify numerical and categorical features
    numerical_features = [col for col in dataset.select_dtypes(exclude=['object']).columns if col not in passthrough_columns]
    categorical_features = [col for col in dataset.select_dtypes(include=['object']).columns if col not in passthrough_columns]


    # Create transformers for numeric and categorical features
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    # Combine all transformers
    preprocessing = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            ('passthrough', 'passthrough', passthrough_columns)
        ]
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessing)
    ])

    return pipeline