import pandas as pd
from zenml import step
from typing_extensions import Annotated

@step(enable_cache=False)
def loading_data(filename: str) -> Annotated[pd.DataFrame,"input_data"]:
    """
    Loads movie data from a CSV file and sets the 'id' column as the index.

    Parameters:
    filename (str): Path to the CSV file.

    Returns:
    Annotated[pd.DataFrame, "input_data"]: DataFrame with weather data, indexed by date.
    """
    data = pd.read_csv(filename,index_col="id")
    return data