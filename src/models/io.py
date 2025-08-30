import json, joblib
from pathlib import Path

def save_model(pipe, model_path: Path):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)

def load_model(model_path: Path):
    return joblib.load(model_path)

def save_schema(schema: dict, schema_path: Path):
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
