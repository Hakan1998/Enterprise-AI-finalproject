from steps import load_movie_data,clean_movie_data,load_rating_data,preprocess_rating_data,merged_data,split_data,create_preprocessing_pipeline,feature_preprocessor
from zenml import pipeline

"""
This pipeline will perform feature engineering on our movie and rating dataset. we combine the two dataset only with valid movieID

Take a look at the create_preprocessing_pipeline step and the feature_preprocessor step. 
You will need to fix some parts there. 
"""
@pipeline
def feature_engineering_pipeline():
    """"
        Pipeline function for performing feature engineering combined data of movie data
        and user data.
    """
    raw_movies = load_movie_data("./data/movies_metadata.csv")
    movies = clean_movie_data(raw_movies)
    raw_users = load_rating_data("./data/ratings_small.csv")
    users = preprocess_rating_data(raw_users)

    dataset = merged_data(movies,users)

    train_data,test_data = split_data(dataset)
    pipeline = create_preprocessing_pipeline(dataset)
    train_data,test_data,pipeline = feature_preprocessor(pipeline,train_data,test_data)

