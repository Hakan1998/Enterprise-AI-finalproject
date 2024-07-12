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

--> Collaborative Filter from Surprise Library
- SVD (Singular Value Decomposition)
- KNN (K-Nearest Neighbors)
- BaselineOnly
- NormalPredictor
- NMF (Non-negative Matrix Factorization)
- SlopeOne
--> Content Based Filter
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

   **utilitys**
   - contains the Email alert logic

### Email Notifications

The project includes email notifications to keep the team informed about the pipeline execution status. Emails are sent using the `smtplib` library.

## Setup and Usage

Is described in main. Just run !pyton run.py and the pipelines will start. 

Further you can you can use the ZenMl and Mflow Dashboard to got more insights. 



### Prerequisites

- Python 3.8 or higher
- ZenML
- Surprise library
- Pandas
- scikit-learn
- smtplib for email notifications

### Limitations
1. **Data Sparsity & Cold starter**: Usual the dataset used has a significant level of sparsity, which can affect the accuracy of recommendations. Since we used an fix csv file this was not problem in our case. But in real world application the possibility of sparse data is high. Further new users or items with no historical data pose a challenge for the recommendation system.
2. **Scalability**: While ZenML provides scalability, the current implementation may require optimization to handle large-scale datasets effectively.
3. **Model Generalization**: The trained models might not generalize well to different datasets without further tuning and validation. We only use the basics Models here, there are many other state-of-art models which would receive way better results. Also our Hyperparamter range here is low, since we dont have much computional resources.

4. **Missing Functionalities**: To make the whole project more realistic, the following points should be done:

    - Create an API instead of loading data from an excel file
    - retrain the Model after we have enough data or a certain time has passed and create some update/deploy rules
    - and many more things
    ...


### Recommendations
1. **Data Augmentation**: Consider incorporating additional data sources to enrich the dataset and reduce sparsity.
2. **Model Optimization**: Explore advanced optimization techniques and distributed training to enhance scalability.
3. **Hybrid Approaches**: Combine collaborative filtering with content-based methods to mitigate the cold start problem.
4. **Continuous Evaluation**: Implement a continuous evaluation pipeline to regularly assess model performance and retrain as necessary.
