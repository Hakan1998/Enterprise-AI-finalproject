from pipelines.training_pipeline import training_pipeline
from pipelines.feature_engineering_pipeline import feature_engineering_pipeline

if __name__ == "__main__":
    feature_engineering_pipeline()
    training_pipeline()