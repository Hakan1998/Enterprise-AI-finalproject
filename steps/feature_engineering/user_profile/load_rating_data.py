import pandas as pd
from zenml import step
from typing_extensions import Annotated

@step(enable_cache=False)
def load_rating_data(filename: str) -> Annotated[pd.DataFrame, "rating_data"]:
    """
    Loads rating data from a CSV file.

    Parameters:
    filename (str): Path to the CSV file.

    Returns:
    Annotated[pd.DataFrame, "rating_data"]: DataFrame containing user ratings.
    """
    ratings = pd.read_csv(filename)
    return ratings
