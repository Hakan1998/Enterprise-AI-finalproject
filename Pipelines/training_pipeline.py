from zenml import pipeline
from zenml.client import Client
from steps import train_baseline_model, train_collaborative_model, train_content_based_model, train_matrix_factorization_model
from evaluate_model import evaluate_model

@pipeline(enable_cache=False)
def training_pipeline():
    """
    Executes the training pipeline.

    This function retrieves preprocessed data from a client, trains multiple models using the training data,
    and evaluates the models using the test data.
    """
    client = Client()
    X_train = client.get_artifact_version("X_train_preprocessed")
    X_test = client.get_artifact_version("X_test_preprocessed")
    y_train = client.get_artifact_version("y_train")
    y_test = client.get_artifact_version("y_test")

    # Train and evaluate Baseline Model
    baseline_model = train_baseline_model(X_train, y_train)
    baseline_metrics = evaluate_model(baseline_model, X_test, y_test, model_type="baseline")
    
    # Train and evaluate Collaborative Filtering Model
    collaborative_model = train_collaborative_model(X_train, y_train)
    collaborative_metrics = evaluate_model(collaborative_model, X_test, y_test, model_type="collaborative")
    
    # Train and evaluate Content-Based Filtering Model
    content_based_model = train_content_based_model(X_train, y_train)
    content_based_metrics = evaluate_model(content_based_model, X_test, y_test, model_type="content_based")
    
    # Train and evaluate Matrix Factorization Model
    matrix_factorization_model = train_matrix_factorization_model(X_train, y_train)
    matrix_factorization_metrics = evaluate_model(matrix_factorization_model, X_test, y_test, model_type="matrix_factorization")

    return {
        "baseline": baseline_metrics,
        "collaborative": collaborative_metrics,
        "content_based": content_based_metrics,
        "matrix_factorization": matrix_factorization_metrics
    }