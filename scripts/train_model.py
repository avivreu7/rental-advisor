# --- make src importable no matter where you run from ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# --------------------------------------------------------

from src.config import CLEANED, MODEL_PKL, SCHEMA_JSON
from src.models.train import train_model
from src.models.io import save_model, save_schema
import pandas as pd
import argparse

def main():
    ap = argparse.ArgumentParser(description="Train a pricing model and export artifacts.")
    ap.add_argument("--model", default="rf", choices=["rf", "xgb", "lgbm", "cat"],
                    help="Which model to train (RandomForest, XGBoost, LightGBM, CatBoost)")
    args = ap.parse_args()

    work = pd.read_csv(CLEANED)
    pipe, meta = train_model(work, model_type=args.model)

    save_model(pipe, MODEL_PKL)
    save_schema({
        "categorical_features": meta["categorical_features"],
        "numeric_features": meta["numeric_features"],
        "target": meta["target"]
    }, SCHEMA_JSON)

    print(f"Model saved to {MODEL_PKL}")
    print(f"Metrics — MAE: {meta['mae']:.2f} | R²: {meta['r2']:.3f}")

if __name__ == "__main__":
    main()
