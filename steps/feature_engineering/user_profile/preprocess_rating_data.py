import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from zenml.steps import step, Output
from datetime import datetime

# Helper function to extract and one-hot encode genres from a string of lists of dictionaries
def extract_and_encode_genres(data, genre_data):
    mlb = MultiLabelBinarizer()
    data = data.apply(lambda x: [i['name'] for i in eval(x) if 'name' in i])
    return pd.DataFrame(mlb.fit_transform(data), columns=mlb.classes_, index=genre_data.index)

@step
def preprocess_rating_data(ratings: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """
    Process rating data by converting timestamps and computing user-specific statistics.
    
    Parameters:
    ratings (pd.DataFrame): DataFrame containing user ratings.
    movies (pd.DataFrame): DataFrame containing movie details including genres.

    Returns:
    pd.DataFrame: Processed ratings with user statistics and genre preferences.
    """
    # Convert timestamp to datetime
    ratings['date'] = pd.to_datetime(ratings['timestamp'], unit='s')
    
    # Merge ratings with movies to get genre information
    ratings = ratings.merge(data[['id', 'genres']], left_on='movieId', right_on='id', how='left')
    
    # Extract and one-hot encode genres
    if 'genres' in ratings.columns:
        ratings = pd.concat([ratings, extract_and_encode_genres(ratings['genres'], ratings)], axis=1)
        ratings.drop(columns=['genres'], inplace=True)
    
    # Compute user statistics
    user_stats = ratings.groupby('userId').agg(
        average_rating=pd.NamedAgg(column='rating', aggfunc='mean'),
        rating_count=pd.NamedAgg(column='rating', aggfunc='count')
    ).reset_index()
    
    # Merge user statistics back to ratings
    processed_ratings = ratings.merge(user_stats, on='userId')
    
    return processed_ratings
