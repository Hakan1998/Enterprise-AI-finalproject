from zenml import step
from zenml.client import Client
import pandas as pd
from typing import Tuple

@step
def retrieve_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retrieves training and testing data from the ZenML client.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The training and testing datasets.
    """
    client = Client()
    
    # Retrieve artifacts by their logged names
    try:
        train_data_artifact = client.get_artifact(name="train_data")
        test_data_artifact = client.get_artifact(name="test_data")
    except KeyError as e:
        raise KeyError(f"Artifact not found: {e}")

    # Ensure the artifacts are in DataFrame format
    train_data = train_data_artifact.read()
    test_data = test_data_artifact.read()

    return train_data, test_data
