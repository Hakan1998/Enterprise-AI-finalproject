from typing import List, Dict, Any, Tuple
from surprise import Trainset
import pandas as pd
from zenml import step
from surprise import Dataset, Reader
from surprise import AlgoBase



def evaluate_model_predictions(model: Any, test_data: List[Tuple], k: int = 10) -> Tuple[float, float, float, float]:
    predictions = model.predict(test_data)
    return predictions

@step
def make_predictions(model: Any, raw_test_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate predictions for a given model and dataset.

    Parameters:
    model (Any): The trained recommendation model.
    raw_test_data (pd.DataFrame): The dataset containing userId, id, and rating columns, plus any additional columns.

    Returns:
    pd.DataFrame: A DataFrame containing userId, id, and the predicted rating.
    """
    # Extract the relevant columns to create test data tuples
    test_data_tuples = [(d['userId'], d['id'], d['rating']) for d in raw_test_data.to_dict(orient='records')]
    evaluate_model_predictions(model, test_data_tuples, k=10)

    
    # Convert the test data tuples to a DataFrame
    test_data_df = pd.DataFrame(test_data_tuples, columns=['userId', 'id', 'rating'])

    # Generate predictions
    predictions = []
    if isinstance(model, AlgoBase):
        # If model is a surprise model
        for _, row in test_data_df.iterrows():
            uid = row['userId']
            iid = row['id']
            pred = model.predict(uid, iid)
            predictions.append(pred.est)
    else:
        # Assume model is an MLflow model
        input_data = test_data_df[['userId', 'id']]
        predictions = model.predict(input_data)
        predictions = predictions.tolist() if hasattr(predictions, 'tolist') else predictions

    # Add the predictions to the DataFrame
    test_data_df['predicted_rating'] = predictions
    return test_data_df