from zenml import step
import mlflow
from typing import Any
from mlflow.exceptions import MlflowException
import pandas as pd
from surprise import AlgoBase
import pickle
import os

"""
Hint:
The original load_best_model function failed because it tried to use mlflow.surprise.load_model, which doesn't exist in MLflow, and had issues handling model files.
We fixed it by changing the function to download the whole model folder with mlflow.artifacts.download_artifacts. 
Then, we load the model from a model.pkl file in that folder using Python's pickle module. 
This ensures the model loads correctly, even if we use the latest version instead of the production version.


"""


class SurpriseModelWrapper(mlflow.pyfunc.PythonModel):
    """
    A custom MLflow Python model wrapper for Surprise recommendation models.

    This class provides a way to wrap a Surprise recommendation model for use 
    with MLflow's model management and deployment functionalities. The wrapper 
    includes methods for loading the model and making predictions.

    Methods:
    --------
    load_context(context):
        Loads the Surprise model from the provided context.
    
    predict(context, model_input):
        Makes predictions using the loaded Surprise model for the given input data.
    """

    def load_context(self, context):
        """
        Load the Surprise model from the MLflow context.

        This method is called by MLflow when the model is deployed or used for 
        inference. It loads the model from the file specified in the context.

        Parameters:
        -----------
        context : mlflow.pyfunc.PythonModelContext
            The context provided by MLflow, which includes the path to the 
            serialized model artifact.
        """
        with open(context.artifacts["model"], 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input):
        """
        Make predictions using the loaded Surprise model.

        This method uses the loaded Surprise model to make predictions for the 
        given input data.

        Parameters:
        -----------
        context : mlflow.pyfunc.PythonModelContext
            The context provided by MLflow, not used in this method.
        
        model_input : pd.DataFrame
            The input data for making predictions. The DataFrame should contain 
            columns 'userId' and 'id'.

        Returns:
        --------
        pd.Series
            A series containing the predicted ratings.
        """
        predictions = []
        for _, row in model_input.iterrows():
            uid = str(row['userId'])
            iid = str(row['id'])
            prediction = self.model.predict(uid, iid)
            predictions.append(prediction.est)
        return pd.Series(predictions)

@step
def load_best_model() -> AlgoBase:
    """
    Load the best model from MLflow as a Surprise model.

    This step loads the best model from the MLflow model registry as a 
    Surprise model. The process involves the following steps:

    1. Download the model directory:
       - Instead of loading the model directly, the entire directory 
         containing the model artifacts is downloaded.
    
    2. Load the model from the directory:
       - The model is loaded from the 'model.pkl' file within the 
         downloaded directory.

    If the model in the 'production' stage is not found, the function 
    attempts to load the latest version of the model from the registry.

    Returns:
    --------
    AlgoBase
        The loaded Surprise model.

    Raises:
    -------
    MlflowException
        If no versions of the model are found in the MLflow registry.
    """
    
    model_uri = "models:/best_model/production"
    try:
        model_dir = mlflow.artifacts.download_artifacts(model_uri)
        model_path = os.path.join(model_dir, "model.pkl")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except MlflowException as e:
        print(f"Error loading production model: {e}. Trying to load the latest model version instead.")
        model_name = "best_model"
        latest_version = mlflow.MlflowClient().get_latest_versions(model_name)
        if latest_version:
            latest_model_uri = f"models:/{model_name}/{latest_version[0].version}"
            model_dir = mlflow.artifacts.download_artifacts(latest_model_uri)
            model_path = os.path.join(model_dir, "model.pkl")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        else:
            raise MlflowException("No versions of model 'best_model' found.")
