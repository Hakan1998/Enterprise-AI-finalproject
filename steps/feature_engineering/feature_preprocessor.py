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
    num_features = train_data[['popularity', 'revenue', 'runtime', 'vote_average', 'vote_count',"user_average_rating","user_rating_count","rating","weighted_rating","user_average_rating","user_rating_count"]]
# Categorical features
    categorical_features = train_data[['original_language', 'status']]
    cat_features_after_encoding = pipeline.named_steps['preprocessing'].transformers_[1][1].named_steps['encoder'].get_feature_names_out(categorical_features)
# Complex features
    complex_features = ['genres', 'production_companies', 'production_countries']
    complex_features_after_encoding = []
    for feature in complex_features:
        complex_feature_names = pipeline.named_steps['preprocessing'].transformers_[2][1].named_steps['extractor_and_binarizer'].mlb.classes_
        complex_features_after_encoding.extend([f"{feature}_{name}" for name in complex_feature_names])
# Text features (assuming TruncatedSVD is used for dimensionality reduction)
    text_features_after_encoding = [f"text_{i}" for i in range(100)]  # Adjust the range based on n_components in TruncatedSVD
# Combine all feature names
    all_features = list(num_features) + list(cat_features_after_encoding) + list(complex_features_after_encoding) + list(text_features_after_encoding)

    train_data_df = pd.DataFrame(train_data_transformed, columns=all_features)
    test_data_df = pd.DataFrame(test_data_transformed, columns=all_features)
    
    return train_data_df, test_data_df, pipeline