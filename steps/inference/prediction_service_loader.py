from zenml import step
import joblib

@step
def prediction_service_loader(model_name: str):
    """
    Loads the model deployment service.
    """
    model_path = f"{model_name}.joblib"
    model = joblib.load(model_path)
    return model