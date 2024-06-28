import pandas as pd
from zenml import step
from typing_extensions import Annotated

@step(enable_cache=False)
def load_inference_data(filename: str) -> Annotated[pd.DataFrame, "new_ratings"]:
    """
    Loads inference rating data from a CSV file.

    Parameters:
    filename (str): Path to the CSV file.

    Returns:
    Annotated[pd.DataFrame, "rating_data"]: DataFrame containing user ratings.
    """
    new_ratings = pd.read_csv(filename)
    #same handlung like feature engineering preprocess_rating_data step
    # drop timestamp column
    new_ratings.drop(columns=['timestamp'], inplace=True)
    
    # Compute user statistics
    user_stats = new_ratings.groupby('userId').agg(
        user_average_rating=pd.NamedAgg(column='rating', aggfunc='mean'),
        user_rating_count=pd.NamedAgg(column='rating', aggfunc='count')
    )
    
    # Merge user statistics back to ratings
    new_ratings = new_ratings.merge(user_stats, on='userId')
    return new_ratings
