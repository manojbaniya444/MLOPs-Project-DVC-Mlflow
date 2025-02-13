## Initialize `dvc` for building different stages

Building Machine Learning Pipeline with dvc at [DVC yaml](./dvc.yaml)

```bash
python -m dvc init

python -m dvc stage add -n data_collection -d src/data_collection.py -o data/raw python src/data_collection.py

python -m dvc repro

python -m dvc dag

python -m dvc metrics show
```

## API for model inference

FastAPI endpoint [FastAPI App](./v0/src/main.py)

```bash
uvicorn main:app --reload --port 8080
```

## Efficient Parameter Tuning

Tuning parameter at one file i.e `params.yaml` at [Parameter Tuning](./params.yaml)

## Git Tag

```shell
git tag -a v1.0 -m "Release V1"
```

## Adding the temp path for data versoning folder

```bash
dvc remote add -d myremote TEMP_PATH_FOLDER

dvc status # to check the status of data and pipeline

dvc checkout
```
