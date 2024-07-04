import pandas as pd
from typing_extensions import Annotated
from zenml import step

@step
def merged_data(filename_rating: str, filename_movie: str, clean_data: pd.DataFrame, processed_ratings: pd.DataFrame) -> Annotated[pd.DataFrame, "merged_data"]:
    """
    Merges movie and user dataframes on a common key.
    """
    new_ratings = pd.read_csv(filename_rating)
    movie_data = pd.read_csv(filename_movie)

    merged_data = pd.merge( movie_data,new_ratings, left_on='id', right_on='movieId', how='inner')
    # Drop the 'movieId' column from the merged DataFrame as it is redundant with 'id'
    merged_data.drop(columns='movieId', inplace=True)
    merged_data = merged_data.sample(n=5000, random_state=42)  # Ensure reproducibility with random_state
    inference_data = merged_data

    client = Client()
    preprocessing_pipeline = client.get_artifact_version("pipeline")
    preprocessed_data = inference_preprocessing(preprocessing_pipeline,data)

    return preprocessed_data