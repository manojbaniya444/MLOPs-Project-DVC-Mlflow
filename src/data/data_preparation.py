import numpy as np
import pandas as pd
import os

train_data = pd.read_csv("./data/raw/train.csv")
test_data = pd.read_csv("./data/raw/test.csv")

# load data
def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}:{e}")

# handle missing value by filling the mean value of the column
def fill_missing_with_mean(df):
    for column in df.columns:
        if df[column].isnull().any():
            mean_value = df[column].mean()
            df[column].fillna(mean_value, inplace=True)
            
    return df

# save csv
def save_csv_file(df: pd.DataFrame, filepath: str) -> None:
    try:
        df.to_csv(filepath, index=False)
        
    except Exception as e:
        raise Exception(f"Error saving csv file {filepath}:{e}")

def main():
    raw_data_file = "./data/raw"
    processed_data_file = "./data/processed"
    
    try:
        train_data = load_data(os.path.join(raw_data_file, "train.csv"))
        test_data = load_data(os.path.join(raw_data_file, "test.csv"))

        train_processed_data = fill_missing_with_mean(train_data)
        test_processed_data = fill_missing_with_mean(test_data)
        
        if not os.path.exists(processed_data_file):
            os.makedirs(processed_data_file)
        
        save_csv_file(train_processed_data, os.path.join(processed_data_file, "train_processed_mean.csv"))
        save_csv_file(test_processed_data, os.path.join(processed_data_file, "test_processed_mean.csv"))
        
    except Exception as e:
        raise Exception(f"Error on data preparation {e}")
    
if __name__ == "__main__":
    main()