import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train_model(work: pd.DataFrame, model_type: str = "rf"):
    X = work.drop(columns=["price"])
    y = work["price"].astype(float)

    cat = [c for c in X.columns if X[c].dtype=="object"]
    num = [c for c in X.columns if c not in cat]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ("num", "passthrough", num)
    ])

    if model_type == "rf":
        model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    elif model_type == "xgb":
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=600, learning_rate=0.06, max_depth=8, subsample=0.9, colsample_bytree=0.8, random_state=42
        )
    elif model_type == "lgbm":
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(n_estimators=800, learning_rate=0.05, max_depth=-1, subsample=0.9, colsample_bytree=0.8, random_state=42)
    elif model_type == "cat":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(depth=8, learning_rate=0.05, n_estimators=800, verbose=False, random_state=42)
    else:
        raise ValueError("unknown model_type")

    pipe = Pipeline([("preprocess", pre), ("model", model)])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    pipe.fit(Xtr, ytr)

    preds = pipe.predict(Xte)
    return pipe, {
        "mae": float(mean_absolute_error(yte, preds)),
        "r2":  float(r2_score(yte, preds)),
        "categorical_features": cat,
        "numeric_features": num,
        "target": "price"
    }
