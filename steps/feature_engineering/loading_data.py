from zenml import step
import pandas as pd
import numpy as np
from typing import Tuple
from typing_extensions import Annotated
from sklearn.model_selection import train_test_split

@step
def loading_data() -> Tuple[Annotated[pd.DataFrame, "train_data"], Annotated[pd.DataFrame, "test_data"]]:
    # Define the number of records
    num_records = 100

    # Create the dataset
    data = pd.DataFrame({
        'userId': np.random.randint(1, 10, num_records),
        'movieId': np.random.randint(1, 100, num_records),
        'rating': np.random.uniform(1, 5, num_records).round(1),
        'timestamp': np.random.randint(1, 1000000000, num_records),
        'budget': np.random.randint(1, 1000000, num_records),
        'genres': np.random.choice(['Action', 'Comedy', 'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance', 'Thriller'], num_records),
        'director': np.random.choice(['Director A', 'Director B', 'Director C', 'Director D'], num_records),
        'actors': np.random.choice(['Actor A', 'Actor B', 'Actor C', 'Actor D'], num_records)
    })

    train_data, test_data = train_test_split(data, test_size=0.25)

    return train_data, test_data
