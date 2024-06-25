from zenml import step
from sklearn.pipeline import Pipeline
import pandas as pd
from typing_extensions import Annotated
from typing import Tuple

@step
def feature_preprocessor(pipeline:Pipeline,X_train:pd.DataFrame,X_test:pd.DataFrame)-> Tuple[Annotated[pd.DataFrame,"X_train_preprocessed"],Annotated[pd.DataFrame,"X_test_preprocessed"],Annotated[Pipeline,"pipeline"]]:
    """Uses the preprocessing pipeline to transform the training and testing datasets.
    """
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed= pipeline.transform(X_test)
    #cat_features_after_encoding = pipeline.named_steps['preprocessing'].transformers_[1][1].named_steps['encoder'].get_feature_names_out(X_train.select_dtypes(include=['object']).columns)
    #all_features = list(X_train.select_dtypes(exclude=['object']).columns )+ list(cat_features_after_encoding)
    X_train_df = pd.DataFrame(X_train_transformed, columns=all_features)
    X_test_df = pd.DataFrame(X_test_transformed, columns=all_features)
    
    return X_train_df, X_test_df, pipeline