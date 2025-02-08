import numpy as np
import pandas as pd
import os
import yaml
import pickle
from sklearn.ensemble import RandomForestClassifier

# load parameters
def load_params(params_file_path: str) -> int:
    try:
        with open(params_file_path, "r") as file:
            params = yaml.safe_load(file)
        return params["model_building"]["n_estimators"]
    except Exception as e:
        raise Exception(f"Error loading parameter in model building {params_file_path}:{e}")
    
# load data
def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}:{e}")
    
# prepare data
def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X_train = data.drop(columns=["Potability"], axis=1)
        y_train = data["Potability"]
        return X_train, y_train
    except Exception as e:
        raise Exception(f"Error preparing data:{e}")
    
# Train model
def train_model(X: pd.DataFrame, y: pd.Series, n_estimators: int) -> RandomForestClassifier:
    try:
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(X, y)
        return clf
    except Exception as e:
        raise Exception(f"Error Training model:{e}")

# save model
def save_model(model: RandomForestClassifier, filepath: str) -> None:
    try:
        with open(filepath, "wb") as file:
            pickle.dump(model, file)
    except Exception as e:
        raise Exception(f"Error saving model {filepath}:{e}")
    
def main():
    params_path = "params.yaml"
    data_path = "./data/processed/train_processed.csv"
    model_name = "models/model.pkl"
    
    try:
        n_estimators = load_params(params_file_path=params_path)
        train_data = load_data(filepath=data_path)
        X_train, y_train = prepare_data(train_data)
        
        model = train_model(X=X_train, y=y_train, n_estimators=n_estimators)
        save_model(model, model_name)
    except Exception as e:
        raise Exception(f"Error on model building {e}")
    
if __name__ == "__main__":
    main()