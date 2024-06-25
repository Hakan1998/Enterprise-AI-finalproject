import pandas as pd
from zenml import step
from typing_extensions import Annotated
from sklearn.preprocessing import LabelEncoder
from typing import Tuple

@step
def label_encoding(y_train:pd.Series,y_test:pd.Series) -> Tuple[Annotated[pd.Series,"y_train_encoded"],Annotated[pd.Series,"y_test_encoded"]]:
    """
    Applies label encoding to the target variable for both training and testing datasets.


    """
    encoder = LabelEncoder()
    y_train = pd.Series(encoder.fit_transform(y_train))
    y_test = pd.Series(encoder.transform(y_test))
    return y_train, y_test