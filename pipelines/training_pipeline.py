from zenml import pipeline
from steps.training.hp_tuner import hp_tuner
from steps.training.model_trainer import model_trainer
from steps.training.evaluate_model import evaluate_model
from steps.training.convert_to_surprise_format import convert_to_surprise_format
from zenml.client import Client

@pipeline(enable_cache=False)
def training_pipeline():
    client = Client()
    train_data = client.get_artifact_version("train_data_preprocessed")
    test_data = client.get_artifact_version("test_data_preprocessed")
    raw_train_data = train_data
    raw_test_data = test_data

    dataset, trainset, test_data = convert_to_surprise_format(raw_train_data=raw_train_data, raw_test_data=raw_test_data)
    best_params_svd, best_params_knn, best_params_baseline, content_model_params = hp_tuner(dataset=dataset, raw_train_data=raw_train_data)
    svd_model, knn_model, baseline_model, content_model = model_trainer(
        
        train_data=trainset, 
        raw_train_data=raw_train_data,
        best_params_svd=best_params_svd, 
        best_params_knn=best_params_knn, 
        best_params_baseline=best_params_baseline,
        content_model_params=content_model_params
    )
    evaluate_model(
        svd_model=svd_model, 
        knn_model=knn_model, 
        baseline_model=baseline_model, 
        content_model=content_model, 
        raw_test_data=raw_test_data
    )

