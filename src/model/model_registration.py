import json
from mlflow.tracking import MlflowClient
import mlflow

import dagshub

# setting the tracking uri for the mlflow
mlflow.set_tracking_uri("https://dagshub.com/manojbaniya727/water-potability-prediction.mlflow")

# initializing the dagshub  repo
dagshub.init(repo_owner='manojbaniya727', repo_name='water-potability-prediction',mlflow=True)

# setting experiment name
mlflow.set_experiment("Model Registration DVC Pipeline test")

# Load the run ID and model name from the json file
reports_path = "reports/run_info.json"
with open(reports_path, "r") as file:
    run_info = json.load(file)
    
run_id = run_info["run_id"]
model_name = run_info["model_name"]

# create an MLFlow client
client = MlflowClient()

# create the model uri to register model in mlflow
model_uri = f"runs:/{run_id}/artifacts/{model_name}"

# register the model
reg = mlflow.register_model(model_uri, model_name)

# get the model version
model_version = reg.version

# transition the model version to staging
new_stage  = "staging"

client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage=new_stage,
    archive_existing_versions=True
)

print(f"Model {model_name} version {model_version} transitioned to {new_stage} stage.")