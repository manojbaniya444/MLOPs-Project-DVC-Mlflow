# import necessary library
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

import mlflow
from mlflow.models import infer_signature
import dagshub

# variables
data_path = r"D:\\MLOP Projects\Water Potability Prediction\water_potability.csv"

# setting the tracking uri for the mlflow
mlflow.set_tracking_uri("https://dagshub.com/manojbaniya727/water-potability-prediction.mlflow")

# initializing the dagshub  repo
dagshub.init(repo_owner='manojbaniya727', repo_name='water-potability-prediction',mlflow=True)

# setting experiment name
mlflow.set_experiment("Experiment 4")

data = pd.read_csv(data_path)

# preparing train and test data
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# fill missing values with mean of each column
def fill_missing_with_mean(df):
    for column in df.columns:
        if df[column].isnull().any():
            medain_value = df[column].mean()
            df[column].fillna(medain_value, inplace=True)
    return df

# apply data preprocessing
train_processed_data = fill_missing_with_mean(train_data)
test_processed_data = fill_missing_with_mean(test_data)

# prepare the train data
X_train = train_processed_data.drop(columns=["Potability"], axis=1)
y_train = train_processed_data["Potability"]

# Define the Random Forest Classifier model and the parameter distribution for hyperparameter tuning
rf = RandomForestClassifier(random_state=42)
param_dist = {
    "n_estimators": [100, 200, 300, 500, 1000],
    "max_depth": [None, 4, 5, 6, 10]
}

# Perform Randomized SearchCV to find the best hyperparameters for the Random Forest Model
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=50, cv=5, n_jobs=-1, verbose=2)

# Start a new Mlflow run to log the random forest tuning process
with mlflow.start_run(run_name="Random-Forest-Tuning") as parent_run:
    # fit the randomized search cv object on the training data to identify best parameter
    random_search.fit(X_train, y_train)
    
    # log the parameter and mean test scores for each combination
    for i in range(len(random_search.cv_results_["params"])):
        with mlflow.start_run(run_name=f"Combination{i+1}", nested=True) as child_run:
            mlflow.log_params(random_search.cv_results_["params"][i])
            mlflow.log_metric("mean_test_score", random_search.cv_results_["mean_test_score"][i])
            
    # print the best hyperparameters found by RandomizedSearchCV
    print("Best parameters found: ", random_search.best_params_)
    
    # Log the best parameters in MLFLow
    mlflow.log_params(random_search.best_params_)

    # train the model using the best parameters identified by RandomizedSearchCV
    best_rf = random_search.best_estimator_
    best_rf.fit(X_train, y_train)

    # save the trained model to a file for later use
    pickle.dump(best_rf, open("model.pkl", "wb"))
    
    # prepare the test data by separating features and target variable
    X_test = test_processed_data.drop(columns=["Potability"], axis=1)
    y_test = test_processed_data["Potability"]
    
    # load the saved model from the file
    model = pickle.load(open("model.pkl", "rb"))
    
    # make predictions on the test using the loaded model
    y_pred = model.predict(X_test)
    
    # calculate and print performance metrics: accuracy, precision, recall, and f1-score
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # log the performance metrics into MLFlow for tracking
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1-score", f1)
    
    mlflow.log_artifact(__file__)

    # Infer the model signature using the test features and predictions
    sign = infer_signature(X_test, random_search.best_estimator_.predict(X_test))
    
    # log the trained model in MLFLOW with its signature
    mlflow.sklearn.log_model(random_search.best_estimator_, "BEST MODEL", signature=sign)
    
    # Print the calculated performance metrics to the console for review
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-Score: ", f1)