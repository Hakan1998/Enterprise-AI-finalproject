from surprise import Dataset, Reader, Trainset
from typing import Tuple, Annotated
import pandas as pd
from zenml import step

@step(enable_cache=False)
def convert_to_surprise_format(
    raw_train_data: pd.DataFrame, 
    raw_test_data: pd.DataFrame
) -> Tuple[
    Annotated[Dataset, "Surprise Dataset"], 
    Annotated[Trainset, "Surprise Trainset"], 
    Annotated[pd.Series, "Test Data"]
]:
    """
    Convert raw training and test data into the Surprise format for collaborative filtering models.

    The Surprise library is used for building and evaluating recommendation systems. 
    It requires data to be in a specific format to train and test collaborative filtering models. 
    This function converts raw training and test data into the required Surprise format.

    Args:
        raw_train_data (pd.DataFrame): The raw training data containing 'userId', 'id', and 'rating' columns.
        raw_test_data (pd.DataFrame): The raw test data containing 'userId', 'id', and 'rating' columns.

    Returns:
        Tuple[Dataset, Trainset, pd.Series]:
            - Dataset: The Surprise dataset object which includes all the data required for training.
            - Trainset: The Surprise trainset object used for training models. This is a processed version of the dataset.
            - pd.Series: The test data converted to a Series of dictionaries, which will be used for making predictions and evaluating the model.
    """
    # Define the rating scale and create a Surprise Reader object
    reader = Reader(rating_scale=(1, 5))
    
    # Load the raw training data into a Surprise Dataset
    # The Dataset object is the main data structure that Surprise uses to handle the data.
    # It reads the data from a pandas DataFrame containing user-item interactions.
    dataset = Dataset.load_from_df(raw_train_data[['userId', 'id', 'rating']], reader)
    
    # Build the full trainset from the dataset
    # The Trainset object is a more processed version of the Dataset that Surprise uses to train models.
    trainset = dataset.build_full_trainset()
    
    # Convert raw test data into a Series of dictionaries
    # This format makes it easier to iterate over the test data for predictions.
    test_data_series = pd.Series(raw_test_data.to_dict(orient='records'))
    
    return dataset, trainset, test_data_series
