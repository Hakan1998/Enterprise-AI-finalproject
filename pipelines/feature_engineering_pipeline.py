from zenml import pipeline
from steps.feature_engineering.loading_data import loading_data
from steps.training.convert_to_surprise_format import convert_to_surprise_format

@pipeline
def feature_engineering_pipeline():
    raw_train_data, raw_test_data = loading_data()
    train_data, test_data = convert_to_surprise_format(raw_train_data=raw_train_data, raw_test_data=raw_test_data)
    return train_data, test_data
