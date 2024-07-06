from zenml import step
import mlflow
from typing import Any
from mlflow.exceptions import MlflowException
from typing import Annotated
import pandas as pd


@step
def load_best_model() -> Annotated[Any, "best_model"]:
    """
    Load the best model from MLflow.
    """
    model_uri = "models:/best_model/production"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except MlflowException as e:
        print(f"Error loading production model: {e}. Trying to load the latest model version instead.")
        model_name = "best_model"
        latest_version = mlflow.MlflowClient().get_latest_versions(model_name)
        if latest_version:
            latest_model_uri = f"models:/{model_name}/{latest_version[0].version}"
            model = mlflow.pyfunc.load_model(latest_model_uri)
            return model
        else:
            raise MlflowException("No versions of model 'best_model' found.")

