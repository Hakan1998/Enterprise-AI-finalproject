from zenml import step
from surprise import Dataset, Reader, Trainset
from typing import Tuple
from typing_extensions import Annotated
import pandas as pd


@step
def convert_to_surprise_format(raw_train_data: pd.DataFrame, raw_test_data: pd.DataFrame) -> Tuple[Annotated[Trainset, "train_data"], Annotated[list, "test_data"]]:
    reader = Reader(rating_scale=(1, 5))
    
    # Convert training data
    train_data = Dataset.load_from_df(raw_train_data[['userId', 'movieId', 'rating']], reader).build_full_trainset()
    
    # Convert test data
    test_data = list(raw_test_data[['userId', 'movieId', 'rating']].itertuples(index=False, name=None))
    
    return train_data, test_data
