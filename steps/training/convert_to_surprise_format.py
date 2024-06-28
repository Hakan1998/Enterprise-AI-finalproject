from zenml import step
from surprise import Dataset, Reader, Trainset
from typing import Tuple
import pandas as pd

@step
def convert_to_surprise_format(raw_train_data: pd.DataFrame, raw_test_data: pd.DataFrame) -> Tuple[Dataset, Trainset, pd.Series]:
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(raw_train_data[['userId', 'movieId', 'rating']], reader)
    trainset = dataset.build_full_trainset()
    test_data_series = pd.Series(raw_test_data.to_dict(orient='records'))
    return dataset, trainset, test_data_series