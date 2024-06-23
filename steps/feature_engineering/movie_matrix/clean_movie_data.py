from zenml.steps import step, Output
import pandas as pd
from zenml import step
from typing_extensions import Annotated
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

@step
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean data by dropping unnecessary columns,rows . fill missing data"""
    columns_to_drop = ["adult", "homepage","imdb_id","video","spoken_languages","tagline","poster_path","original_title","belongs_to_collection","release_date"]
    data = data.drop(columns=columns_to_drop, errors='ignore')
    
    
    # Text data
    data['overview'] = data['overview'].fillna('No overview available')
    
    # Text data
    data['title'] = data['title'].fillna('Unknown Title')

    # Dropping rows where either 'production_companies' or 'production_countries' is missing,
    # because they are dictionary list.
    data = data.dropna(subset=['production_companies', 'production_countries'])
    # there are 3 rows with time format data in id column
    data = data[pd.to_numeric(data['id'], errors='coerce').notna()]

    return data