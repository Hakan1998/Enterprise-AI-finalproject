from surprise import Dataset, Reader, Trainset
from typing import Tuple, Annotated, List, Dict
import pandas as pd
from zenml import step

@step(enable_cache=False)
def get_test_data_series(test_data: pd.DataFrame) -> Tuple[Annotated[List[Tuple[int, int, float]], "Test Data"]]:
    # Convert raw test data into a list of dictionaries
    test_data_tuples = [(d['userId'], d['id'], d['rating']) for d in test_data.to_dict(orient='records')]
    
    return test_data_tuples