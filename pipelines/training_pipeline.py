from zenml import pipeline
from steps.training.retrieve_data import retrieve_data
from steps.training.hp_tuner import hp_tuner
from steps.training.model_trainer import model_trainer
from steps.training.evaluate_model import evaluate_model
from steps.feature_engineering.loading_data import loading_data
from sklearn.model_selection import train_test_split

@pipeline
def training_pipeline():
    train_data, test_data = loading_data()
    best_params = hp_tuner(train_data)
    user_cf_recs, item_cf_recs, svd_recs, content_recs = model_trainer(train_data, test_data, best_params)
    results = evaluate_model(train_data, test_data, user_cf_recs, item_cf_recs, svd_recs, content_recs)

