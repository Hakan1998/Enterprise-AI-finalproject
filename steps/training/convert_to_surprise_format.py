from surprise import Dataset, Reader, Trainset
from typing import Tuple, Annotated
import pandas as pd
from zenml import step

@step
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

    Args:
        raw_train_data (pd.DataFrame): The raw training data containing 'userId', 'id', and 'rating' columns.
        raw_test_data (pd.DataFrame): The raw test data containing 'userId', 'id', and 'rating' columns.

    Returns:
        Tuple[Dataset, Trainset, pd.Series]:
            - Dataset: The Surprise dataset object.
            - Trainset: The Surprise trainset object used for training models.
            - pd.Series: The test data converted to a Series of dictionaries.
    """
    # Define the rating scale and create a Surprise Reader object
    reader = Reader(rating_scale=(1, 5))
    
    # Load the raw training data into a Surprise Dataset
    dataset = Dataset.load_from_df(raw_train_data[['userId', 'id', 'rating']], reader)
    
    # Build the full trainset from the dataset
    trainset = dataset.build_full_trainset()
    
    # Convert raw test data into a Series of dictionaries
    test_data_series = pd.Series(raw_test_data.to_dict(orient='records'))
    
    return dataset, trainset, test_data_series