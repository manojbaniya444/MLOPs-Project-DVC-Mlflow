import dagshub
import mlflow

# setting the tracking uri for the mlflow
mlflow.set_tracking_uri("https://dagshub.com/manojbaniya727/water-potability-prediction.mlflow")

# initializing the dagshub  repo
dagshub.init(repo_owner='manojbaniya727', repo_name='water-potability-prediction',mlflow=True)

# set experiment name
mlflow.set_experiment("Test Experiment")

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)