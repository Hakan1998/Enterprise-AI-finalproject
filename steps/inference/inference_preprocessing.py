from zenml import step
import pandas as pd
from sklearn.preprocessing import StandardScaler

@step
def inference_preprocessing(preprocessing_pipeline, data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the inference data.
    """
    scaler = StandardScaler()
    preprocessed_data = scaler.fit_transform(data)
    return pd.DataFrame(preprocessed_data, columns=data.columns)