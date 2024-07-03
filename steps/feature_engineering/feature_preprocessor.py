from zenml import step
from sklearn.pipeline import Pipeline
import pandas as pd
from typing_extensions import Annotated
from typing import Tuple


@step
def feature_preprocessor(
    pipeline: Pipeline, 
    train_data: pd.DataFrame, 
    test_data: pd.DataFrame
) -> Tuple[Annotated[pd.DataFrame, "train_data_preprocessed"], 
           Annotated[pd.DataFrame, "test_data_preprocessed"], 
           Annotated[Pipeline, "pipeline"]]:
    """Uses the preprocessing pipeline to transform the training and testing datasets."""
    # Fit the pipeline to the training data
    pipeline.fit(train_data)

    # Get the feature names after transformation
    preprocessor = pipeline.named_steps['preprocessing']
    feature_names = []
    for name, transformer, columns in preprocessor.transformers:
        if name == 'remainder':
            if transformer == 'passthrough':
                feature_names.extend(columns)
            else:
                continue
        elif name == 'cat':
            onehot_columns = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(columns)
            feature_names.extend(onehot_columns)
        elif name == 'num':
            feature_names.extend(columns)
        elif name == 'passthrough':
            feature_names.extend(columns)

    # Transform the training and testing data
    train_data_transformed = pipeline.transform(train_data)
    test_data_transformed = pipeline.transform(test_data)

    # Convert the transformed data back to DataFrames with new feature names
    train_data_preprocessed = pd.DataFrame(train_data_transformed, columns=feature_names)
    test_data_preprocessed = pd.DataFrame(test_data_transformed, columns=feature_names)
    pd.set_option('display.max_columns', None)
    print(train_data_preprocessed)

    return train_data_preprocessed, test_data_preprocessed, pipeline