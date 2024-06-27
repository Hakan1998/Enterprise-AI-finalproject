from zenml import pipeline
from steps.training.hp_tuner import hp_tuner
from steps.training.model_trainer import model_trainer
from steps.training.evaluate_model import evaluate_model
from steps.training.convert_to_surprise_format import convert_to_surprise_format
from steps.feature_engineering.loading_data import loading_data

@pipeline
def training_pipeline():
    # Load data
    raw_train_data, raw_test_data = loading_data()

    # Convert data to Surprise format
    train_data, test_data = convert_to_surprise_format(raw_train_data=raw_train_data, raw_test_data=raw_test_data)

    # Perform hyperparameter tuning
    best_params_svd, best_params_knn, best_params_baseline = hp_tuner(train_data=train_data)

    # Train models with the best hyperparameters
    svd_model, knn_model, baseline_model = model_trainer(train_data=train_data, best_params_svd=best_params_svd, best_params_knn=best_params_knn, best_params_baseline=best_params_baseline)

    # Evaluate the trained models
    evaluate_model(svd_model=svd_model, knn_model=knn_model, baseline_model=baseline_model, test_data=test_data)
