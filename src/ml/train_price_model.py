import pandas as pd
import joblib
from pathlib import Path
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_CSV = Path("data/ml/property_dataset.csv")
MODEL_PATH = Path("models/price_model.joblib")

def train():
    df = pd.read_csv(DATA_CSV)

    # clean
    df["neighborhood"] = df["neighborhood"].fillna("unknown").astype(str).str.strip().str.lower()
    df["property_type"] = df["property_type"].fillna("unknown").astype(str).str.strip().str.lower()
    df["rooms"] = df["rooms"].astype(int)
    df["size_sqm"] = df["size_sqm"].astype(int)
    df["price_eur"] = df["price_eur"].astype(int)

    numeric = ["rooms", "size_sqm"]
    categorical = ["neighborhood", "property_type"]
    X = df[numeric + categorical].copy()
    y = df["price_eur"].astype(float)

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

    pipe = Pipeline([
        ("pre", pre),
        ("model", Ridge(alpha=1.0)),
    ])

    # ---- Cross-validation (more reliable with tiny data) ----
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mae = -cross_val_score(pipe, X, y, cv=cv, scoring="neg_mean_absolute_error")
    cv_rmse = -cross_val_score(pipe, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    cv_r2 = cross_val_score(pipe, X, y, cv=cv, scoring="r2")

    print("=== Cross-validation (5-fold) ===")
    print(f"Rows total: {len(df)}")
    print(f"MAE:  {cv_mae.mean():,.0f} EUR (+/- {cv_mae.std():,.0f})")
    print(f"RMSE: {cv_rmse.mean():,.0f} EUR (+/- {cv_rmse.std():,.0f})")
    print(f"R2:   {cv_r2.mean():.3f} (+/- {cv_r2.std():.3f})")

    # ---- Single split (optional, for a simple headline metric) ----
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)

    print("\n=== Evaluation (test split) ===")
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"MAE:  {mae:,.0f} EUR")
    print(f"RMSE: {rmse:,.0f} EUR")
    print(f"R2:   {r2:.3f}")

    # Fit on full data for final model artifact
    pipe.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"\nSaved model: {MODEL_PATH}")

if __name__ == "__main__":
    train()