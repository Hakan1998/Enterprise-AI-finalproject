import pandas as pd
from zenml import step
from typing_extensions import Annotated
from sklearn.model_selection import train_test_split
from typing import Tuple


@step
def split_data(dataset:pd.DataFrame, label: str) -> Tuple[
    Annotated[pd.DataFrame,"X_train"],
    Annotated[pd.DataFrame,"X_test"],
    Annotated[pd.Series,"y_train"],
    Annotated[pd.Series,"y_test"]]:
    """
    Splits a dataset into training and testing sets, separating features from the target label.
    """
    X = dataset.drop(label,axis=1)
    Y = dataset[label]
    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,shuffle=False)
    return X_train,X_test,y_train,y_test