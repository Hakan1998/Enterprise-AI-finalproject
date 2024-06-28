from zenml import pipeline
from steps.feature_engineering.loading_data import loading_data
import pandas as pd

@pipeline
def feature_engineering_pipeline():
    loading_data()
