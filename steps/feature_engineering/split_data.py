import pandas as pd
from zenml import step
from typing_extensions import Annotated
from sklearn.model_selection import train_test_split
from typing import Tuple


@step
def split_data(dataset: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "train_data"],
    Annotated[pd.DataFrame, "test_data"]]:
    
    """
    Splits a dataset into training and testing sets.
    """
    train_data, test_data = train_test_split(dataset, test_size=0.2, shuffle=True, random_state=42)
    return train_data, test_data