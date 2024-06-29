import pandas as pd
import numpy as np
from surprise import Dataset, Reader
from zenml import step
from sklearn.model_selection import train_test_split
from typing import Tuple

@step(enable_cache=False)
def loading_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    num_records = 100
    data = pd.DataFrame({
        'userId': np.random.randint(1, 700, num_records),
        'movieId': np.random.randint(1, 900, num_records),
        'rating': np.random.uniform(1, 5, num_records).round(1),
        'timestamp': np.random.randint(1, 1000000000, num_records),
        'budget': np.random.randint(1, 1000000, num_records),
        'genres': np.random.choice(['Action', 'Comedy', 'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance', 'Thriller'], num_records),
        'director': np.random.choice(['Director A', 'Director B', 'Director C', 'Director D'], num_records),
        'actors': np.random.choice(['Actor A', 'Actor B', 'Actor C', 'Actor D'], num_records)
    })
    raw_train_data, raw_test_data = train_test_split(data, test_size=0.25)
    return raw_train_data, raw_test_data