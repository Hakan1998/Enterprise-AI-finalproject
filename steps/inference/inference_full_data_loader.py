import pandas as pd
from typing_extensions import Annotated
from zenml import step

@step
def infernce_full_data_loader(clean_data: pd.DataFrame, new_ratings: pd.DataFrame) -> Annotated[pd.DataFrame, "merged_inference_data"]:
    """
    Merges movie and new ratings dataframes on a common key.
    """
    # Specify the merge keys and method inside the function
    merged_inference_data = pd.merge(clean_data, new_ratings, left_on='id', right_on='movieId', how='inner')
    # Drop the 'movieId' column from the merged DataFrame as it is redundant with 'id'
    merged_inference_data.drop(columns='movieId', inplace=True)

    return merged_inference_data
