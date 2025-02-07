```bash
python -m dvc init

python -m dvc stage add -n data_collection -d src/data_collection.py -o data/raw python src/data_collection.py

python -m dvc repro

python -m dvc dag

python -m dvc metrics show
```

```bash
uvicorn main:app --reload --port 8080
```
