import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns

import pickle

import mlflow
import mlflow.sklearn
import dagshub

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

print("Exp 1")

# variables
data_path = r"D:\\MLOP Projects\Water Potability Prediction\water_potability.csv"

# setting the tracking uri for the mlflow
mlflow.set_tracking_uri("https://dagshub.com/manojbaniya727/water-potability-prediction.mlflow")

# initializing the dagshub  repo
dagshub.init(repo_owner='manojbaniya727', repo_name='water-potability-prediction',mlflow=True)

# Setting Experiment Name
mlflow.set_experiment("Experiment 1")

print("MLFLOW set")

data = pd.read_csv(data_path)


train_data, test_data = train_test_split(data, test_size=0.20, random_state=2)

# util to handle missing values in the dataset with the median value of each column
def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():
            medain_value = df[column].median()
            df[column].fillna(medain_value, inplace=True)
    return df

# fill the missing values in both the training and test datasets using the median
train_processed_data = fill_missing_with_median(train_data)
test_procesed_data = fill_missing_with_median(test_data)

# Train feature and Train Label
X_train = train_processed_data.drop(columns=["Potability"], axis=1)
y_train = train_processed_data["Potability"]

n_estimators = 100

print("Everything done now running")

# Start a new MLFlow run for tracking the experiment
with mlflow.start_run():
    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(X_train, y_train)
    
    print("Model fit successful")

    # save the trained model to a file using pickle
    pickle.dump(clf, open("model.pkl", "wb"))

    # Prepare the test data for prediction 
    X_test = test_procesed_data.drop(columns=["Potability"], axis=1)
    y_test = test_procesed_data["Potability"]
    
    model = pickle.load(open("model.pkl", "rb"))
    
    # get the predictions
    y_pred = model.predict(X_test)
    
    # calculate the performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # log  the metrics to MLFLOW for tracking
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1-Score", f1)
    
    print("Metrics logged to mlflow")
    
    # Generate a confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True)
    plt.xlabel("Prediction")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    plt.savefig("confusion_matrix.png")

    # save the confusion matrix plot as a PNG file
    mlflow.log_artifact("confusion_matrix.png")

    # log the train model to MLFlow
    mlflow.sklearn.log_model(clf, "RandomForestClassifier")
    
    # Log the artifact
    mlflow.log_artifact(__file__)

    # set tags in MLFlow to store additional metadata
    mlflow.set_tag("author", "manoj")
    mlflow.set_tag("model", "GB")
    
    # Print the performance metrics for reference
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-Score: ", f1)