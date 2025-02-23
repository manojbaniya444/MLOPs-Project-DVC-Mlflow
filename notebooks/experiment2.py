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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

print("Exp 2")

# variables
data_path = r"D:\\MLOP Projects\Water Potability Prediction\water_potability.csv"

# setting the tracking uri for the mlflow
mlflow.set_tracking_uri("https://dagshub.com/manojbaniya727/water-potability-prediction.mlflow")

# initializing the dagshub  repo
dagshub.init(repo_owner='manojbaniya727', repo_name='water-potability-prediction',mlflow=True)

# setting experiment name
mlflow.set_experiment("Experiment 2")

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
# Prepare the test data for prediction 
X_test = test_procesed_data.drop(columns=["Potability"], axis=1)
y_test = test_procesed_data["Potability"]
    
# Model Parameters
n_estimators = 100

print("Everything done now running")

# multiple model defining to compare performance
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Classifier": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "XG Boost": XGBClassifier()
}

# Start a new parent MLFlow run to track the overall experiment
with mlflow.start_run(run_name="multiple-classifier-model-experiment"):
    # Iterate over each model in the dictionary
    for model_name, model in models.items():
        # start a child run withing the parent run for each individual model
        with mlflow.start_run(run_name=model_name, nested=True):
            # train the model on the training data
            model.fit(X_train, y_train)
            # save the trained model using pickle
            model_filename = f"{model_name.replace(" ", "_")}.pkl"
            pickle.dump(model, open(model_filename, "wb"))
            
            # performance of model in testing dataset
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
                    
            # Generate a confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5, 5))
            sns.heatmap(cm, annot=True)
            plt.xlabel("Prediction")
            plt.ylabel("True Label")
            plt.title(f"Confusion Matrix for {model_name}")

            plt.savefig(f"confusion_matrix_{model_name.replace(" ", "_")}.png")

            # save the confusion matrix plot as a PNG file
            mlflow.log_artifact(f"confusion_matrix_{model_name.replace(" ", "_")}.png")

            # log the train model to MLFlow
            mlflow.sklearn.log_model(model, model_name.replace(" ", "_"))
            
            # Log the artifact
            mlflow.log_artifact(__file__)

            # set tags in MLFlow to store additional metadata
            mlflow.set_tag("author", "manoj")
            
print("All model have been trained and logged as child run successfully.")