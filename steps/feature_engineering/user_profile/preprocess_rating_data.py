import pandas as pd
from zenml import step


@step
def preprocess_rating_data(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Process rating data by converting timestamps and computing user-specific statistics.
    
    Parameters:
    ratings (pd.DataFrame): DataFrame containing user ratings.
    movies (pd.DataFrame): DataFrame containing movie details including genres.

    Returns:
    pd.DataFrame: Processed ratings with user statistics and genre preferences.
    """
    # drop timestamp column
    ratings.drop(columns=['timestamp'], inplace=True)
    
    # Compute user statistics
    user_stats = ratings.groupby('userId').agg(
        average_rating=pd.NamedAgg(column='rating', aggfunc='mean'),
        rating_count=pd.NamedAgg(column='rating', aggfunc='count')
    )
    
    # Merge user statistics back to ratings
    processed_ratings = ratings.merge(user_stats, on='userId')
    
    return processed_ratings
