import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# =========================
# 1. LOAD DATA
# =========================

df = pd.read_csv(r"C:\Users\moksh\Desktop\bikePrice Prediction\Used_Bikes.csv")

print("Initial Shape:", df.shape)

# =========================
# 2. BASIC CLEANING
# =========================

df = df[df["kms_driven"] < 200000]
df = df[df["age"] < 30]
df = df[df["power"] < 1000]

print("After Cleaning Shape:", df.shape)

# =========================
# 3. ADVANCED FEATURE ENGINEERING
# =========================

# Extract engine CC (2–4 digits only)
df["engine_from_name"] = df["bike_name"].str.extract(r"(\d{2,4})")
df["engine_from_name"] = pd.to_numeric(df["engine_from_name"], errors="coerce")

# Extract model series (second token heuristic)
df["model_series"] = df["bike_name"].str.split().str[1].fillna("unknown")

# Fill missing engine values
df["engine_from_name"] = df["engine_from_name"].fillna(df["power"])

# Create km_per_year feature
df["km_per_year"] = df["kms_driven"] / (df["age"] + 1)

# Group rare cities
TOP_N_CITIES = 50
top_cities = df["city"].value_counts().nlargest(TOP_N_CITIES).index
df["city"] = df["city"].where(df["city"].isin(top_cities), "Other")

# Convert owner to numeric
owner_map = {
    "First Owner": 1,
    "Second Owner": 2,
    "Third Owner": 3,
    "Fourth Owner Or More": 4
}
df["owner"] = df["owner"].map(owner_map)

# Drop original bike_name
df = df.drop("bike_name", axis=1)

# =========================
# 4. DEFINE FEATURES
# =========================

X = df.drop("price", axis=1)
y = df["price"]  # DO NOT log manually (we'll use TransformedTargetRegressor)

categorical_cols = ["brand", "city", "model_series"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# =========================
# 5. PREPROCESSOR
# =========================

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# =========================
# 6. MODEL
# =========================

xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", xgb)
])

# 🔥 Log-transform target automatically
model = TransformedTargetRegressor(
    regressor=pipeline,
    func=np.log1p,
    inverse_func=np.expm1
)

# =========================
# 7. TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 8. BACKUP OLD MODEL (if exists)
# =========================

if os.path.exists("bike_price_model.pkl"):
    os.replace("bike_price_model.pkl", "bike_price_model_v1_backup.pkl")
    print("Old model backed up as bike_price_model_v1_backup.pkl")

# =========================
# 9. TRAIN MODEL
# =========================

print("Training new model...")
model.fit(X_train, y_train)
print("Training completed!")

# =========================
# 10. EVALUATE
# =========================

test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

mae = mean_absolute_error(y_test, test_pred)
r2_test = r2_score(y_test, test_pred)
r2_train = r2_score(y_train, train_pred)

print("\nModel Performance")
print("MAE:", round(mae, 2))
print("Train R2:", round(r2_train, 4))
print("Test R2:", round(r2_test, 4))

# =========================
# 11. SAVE NEW MODEL
# =========================

joblib.dump(model, "bike_price_model.pkl")
print("New model saved as bike_price_model.pkl")