import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml

# Load Parameters
def load_params(filepath: str) -> float:
    try:
        with open(filepath, "r") as file:
            params = yaml.safe_load(file)
        return params["data_collection"]["test_size"]
    except Exception as e:
        raise Exception(f"Error loading parameter from {filepath}:{e}")

# Load data
def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}:{e}")

# Split data
def split_data(data: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        return train_test_split(data, test_size=test_size)
    except ValueError as e:
        raise ValueError(f"Error splitting data:{e}")

# Save CSV
def save_csv(df:pd.DataFrame, filepath:str):
    try:
        df.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f"Error saving csv file {filepath}:{e}")
def main():
    data_filepath = r"D:\\MLOP Projects\Water Potability Prediction\water_potability.csv"
    params_filepath = "params.yaml"
    
    raw_data_path = os.path.join("data", "raw")
    if not os.path.exists(raw_data_path):
        os.makedirs(raw_data_path)
    
    try:
        data = load_data(data_filepath)
        test_size = load_params(params_filepath)
        train_data, test_data = split_data(data, test_size)
        
        save_csv(train_data, os.path.join(raw_data_path, "train.csv"))
        save_csv(test_data, os.path.join(raw_data_path, "test.csv"))
    except Exception as e:
        raise Exception(f"An error occur in data collection:{e}")
    
if __name__ == "__main__":
    main()