from zenml import step
from sklearn.pipeline import Pipeline
import pandas as pd
from typing_extensions import Annotated
from typing import Tuple

@step
def feature_preprocessor(pipeline:Pipeline,train_data:pd.DataFrame,test_data:pd.DataFrame)-> Tuple[Annotated[pd.DataFrame,"train_data_preprocessed"],Annotated[pd.DataFrame,"test_data_preprocessed"],Annotated[Pipeline,"pipeline"]]:
    """Uses the preprocessing pipeline to transform the training and testing datasets.
    """
    train_data_transformed = pipeline.fit_transform(train_data)
    test_data_transformed= pipeline.transform(test_data)
    # Feature names for categorical data

    #cat_features_after_encoding = pipeline.named_steps['preprocessing'].named_transformers_['cat'].get_feature_names_out(['original_language', 'status',"id","userId"])

    # Complex features
    #complex_features = ['genres']
    #complex_features_after_encoding = []
    #for feature in complex_features:
        #complex_feature_names = pipeline.named_steps['preprocessing'].named_transformers_[feature].named_steps['extractor_and_binarizer'].mlb.classes_
        #complex_features_after_encoding.extend([f"{feature}_{name}" for name in complex_feature_names])

    # Text features (assuming TruncatedSVD is used for dimensionality reduction)
    #text_features_after_encoding = [f"text_{i}" for i in range(100)]  # Adjust the range based on n_components in TruncatedSVD

    # Combine all feature names
    cat_features_after_encoding = pipeline.named_steps['preprocessing'].transformers_[1][1].named_steps['encoder'].get_feature_names_out(train_data.select_dtypes(include=['object']).columns)
    all_features = list(train_data.select_dtypes(exclude=['object']).columns )+ list(cat_features_after_encoding)
    
    #list(complex_features_after_encoding)
    # list(text_features_after_encoding)
    train_data_df = pd.DataFrame(train_data_transformed, columns=all_features)
    test_data_df = pd.DataFrame(test_data_transformed, columns=all_features)
    
    return train_data_df, test_data_df, pipeline