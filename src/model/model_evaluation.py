import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dvclive import Live
import yaml

# load_data
def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading file {filepath}:{e}")
    
# prepare data
def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.drop(columns=["Potability"], axis=1)
        y = data["Potability"]
        return X, y
    except Exception as e:
        raise Exception(f"Error preparing data on model evaluation:{e}")

# load model
def load_model(filepath: str):
    try:
        with open(filepath, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        raise Exception(f"Error loading model on model evaluation {filepath}:{e}")

# model evaluation
def evaluation_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    try:
        params = yaml.safe_load(open("params.yaml", "r"))
        test_size = params["data_collection"]["test_size"] 
        n_estimators = params["model_building"]["n_estimators"]
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        with Live(save_dvc_exp=True) as live:
            live.log_metric("accuracy", accuracy)
            live.log_metric("precision", precision)
            live.log_metric("recall", recall)
            live.log_metric("f1_score", f1)
            
            live.log_param("test_size", test_size)
            live.log_param("n_estimators", n_estimators)
        
        metrics_dict = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        return metrics_dict
    except Exception as e:
        raise Exception(f"Error evaulation model:{e}")
    
# save result
def save_metrics(metrics_dict: dict, filepath: str):
    try:
        with open(filepath, "w") as file:
            json.dump(metrics_dict, file, indent=5)
    except Exception as e:
        raise Exception(f"Error saving model result {filepath}:{e}")

def main():
    test_data_path = "./data/processed/test_processed_median.csv"
    model_path = "models/model.pkl"
    metrics_path = "reports/metrics.json"
    
    try:
        test_data = load_data(test_data_path)
        X_test, y_test = prepare_data(test_data)
        model = load_model(model_path)
        evaluation_result = evaluation_model(model, X_test, y_test)
        save_metrics(evaluation_result, metrics_path)
    except Exception as e:
        raise Exception(f"Error on model evaluation:{e}")

if __name__ == "__main__":
    main()