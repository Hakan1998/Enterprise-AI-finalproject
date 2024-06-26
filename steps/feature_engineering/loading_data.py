from zenml import step
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing_extensions import Annotated
from typing import Tuple

@step(enable_cache=False)
def loading_data() -> Tuple[Annotated[pd.DataFrame, "train_data"], Annotated[pd.DataFrame, "test_data"]]:
    """
    Loads synthetic data and splits it into training and testing sets.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The training and testing datasets.
    """
    num_records = 10000
    data = pd.DataFrame({
        'userId': np.random.randint(1, 100, num_records),
        'movieId': np.random.randint(1, 1000, num_records),
        'rating': np.random.uniform(1, 5, num_records).round(1),
        'timestamp': np.random.randint(1, 1000000000, num_records),
        'budget': np.random.randint(1, 1000000, num_records),       
    })

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    return train_data, test_data
