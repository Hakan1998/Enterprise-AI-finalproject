from zenml.steps import step, Output
import pandas as pd
from zenml import step
from typing_extensions import Annotated
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

@step
def clean_movie_data(data: pd.DataFrame) -> Annotated[pd.DataFrame,"clean_data"]:
    """Clean data by dropping unnecessary columns,rows . fill missing data"""
    columns_to_drop = ["adult", "homepage","imdb_id","video","spoken_languages","poster_path","original_title","belongs_to_collection","release_date","production_companies","production_countries","genres"]
    data = data.drop(columns=columns_to_drop, errors='ignore')
    print("der Datentyp ist:")

    print(data['id'].dtype)
    
    # handling text data and trim whitespace
    #data['overview'] = data['overview'].fillna('No overview available').str.lower().str.strip()
    #data['title'] = data['title'].fillna('Unknown Title').str.lower().str.strip()

    # Dropping rows where either 'production_companies' or 'production_countries' is missing,
    # because they are dictionary list.
    #data = data.dropna(subset=['production_companies', 'production_countries'])
    # there are 3 rows with time format data in id column
    data = data[pd.to_numeric(data['id'], errors='coerce').notna()]
    data.dropna(subset=['id'], inplace=True)
    
    # Calculate mean rating across all movies
    C = data['vote_average'].mean()

    # Calculate the minimum number of votes required to be listed (e.g., 90th percentile)
    m = data['vote_count'].quantile(0.90)

    # Function to calculate weighted rating for each row
    def weighted_rating(x, m=m, C=C):
        v = x['vote_count']
        R = x['vote_average']
        return (v / (v + m) * R) + (m / (m + v) * C)

    data['weighted_rating'] = data.apply(weighted_rating, axis=1)

    return data