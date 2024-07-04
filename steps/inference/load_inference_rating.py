import pandas as pd
from zenml import step
from typing_extensions import Annotated


@step
def load_inference_rating(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)
