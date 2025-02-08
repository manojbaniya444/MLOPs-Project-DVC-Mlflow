import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml

data = pd.read_csv(r"D:\\MLOP Projects\Water Potability Prediction\water_potability.csv")

test_size = yaml.safe_load(open("params.yaml"))["data_collection"]["test_size"]

train_data, test_data = train_test_split(data, test_size=test_size, random_state=2)

data_path = os.path.join("data", "raw")

os.makedirs(data_path)

train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)