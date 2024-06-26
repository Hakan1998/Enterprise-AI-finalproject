from zenml import pipeline
from steps.feature_engineering.loading_data import loading_data
from sklearn.model_selection import train_test_split
import pandas as pd

@pipeline
def feature_engineering_pipeline():
    data = loading_data()
    if isinstance(data, pd.DataFrame):
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        print(train_data.head())
        print(test_data.head())
    else:
        print("Error: Data is not a valid pandas DataFrame.")
    return data