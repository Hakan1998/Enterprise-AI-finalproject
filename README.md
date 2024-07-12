# End-to-End Machine Learning Pipeline for Movie Recommendations

## Purpose
This repository contains the implementation of an end-to-end machine learning pipeline developed for a university project. The objective is to build a recommendation system that predicts top K movie matches for each user based on their historical ratings. The pipeline is structured using ZenML to ensure modularity, reproducibility, and scalability.

--> Data Used: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
--> Models Used: 
  Github:  https://github.com/NicolasHug/Surprise?tab=readme-ov-file
  Doku:https://surprise.readthedocs.io/en/stable/index.html


## Project Overview
The project is organized into three main pipelines:

1. **Feature Engineering Pipeline**
2. **Training Pipeline**
3. **Inference Pipeline**

### 1. Feature Engineering Pipeline
This pipeline handles data loading, cleaning, merging, and feature preprocessing tasks. It prepares the data for model training by ensuring it is in a suitable format.

### 2. Training Pipeline
The training pipeline involves:
- Converting data to the Surprise library format
- Hyperparameter tuning for various recommendation algorithms
- Training multiple models with the best parameters
- Evaluating model performance using metrics such as RMSE, MAE, Precision, and Recall

### 3. Inference Pipeline
The inference pipeline:
- Loads the best-trained model from the training pipeline
- Preprocesses new inference data
- Generates top K movie recommendations for each user based on the trained model


### Data

- **Movie Data**: Contains metadata about movies.
- **Rating Data**: Contains user ratings for different movies.


### Models

--> Basicly all Basic prediction Algos provived from the Surprise Libaray
- SVD (Singular Value Decomposition)
- KNN (K-Nearest Neighbors)
- BaselineOnly
- NormalPredictor
- NMF (Non-negative Matrix Factorization)
- SlopeOne
- Content-based Filtering using Cosine Similarity

### Steps


1. **Feature Engineering**:
   - `load_movie_data`: Load movie data from a CSV file.
   - `clean_movie_data`: Clean the movie data.
   - `load_rating_data`: Load rating data from a CSV file.
   - `preprocess_rating_data`: Preprocess the rating data.
   - `merged_data`: Merge movie and rating data.
   - `split_data`: Split data into training and testing sets.
   - `create_preprocessing_pipeline`: Create a preprocessing pipeline.
   - `feature_preprocessor`: Apply feature preprocessing.

2. **Model Training**:
   - `convert_to_surprise_format`: Convert data to Surprise library format.
   - `hp_tuner`: Perform hyperparameter tuning.
   - `model_trainer`: Train multiple models.
   - `evaluate_model`: Evaluate trained models.

3. **Inference**:
   - `load_best_model`: Load the best trained model.
   - `load_and_preprocess_inference_data`: Preprocess new inference data.
   - `make_recommendations`: Generate top K recommendations for each user.

### Email Notifications

The project includes email notifications to keep the team informed about the pipeline execution status. Emails are sent using the `smtplib` library.

## Setup and Usage

### Prerequisites

- Python 3.8 or higher
- ZenML
- Surprise library
- Pandas
- scikit-learn
- smtplib for email notifications

