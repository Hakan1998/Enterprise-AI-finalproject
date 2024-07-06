import pandas as pd
from zenml import step
from typing_extensions import Annotated

@step
def load_movie_data(filename: str) -> pd.DataFrame:
    """
    Load movie data from a CSV file and drop the 'popularity' column.
    """
    # Load data with low_memory=False to avoid DtypeWarning
    data = pd.read_csv(filename, low_memory=False)
    
    # Drop the 'popularity' column
    if 'popularity' in data.columns:
        data.drop(columns=['popularity'], inplace=True)
    
    return data