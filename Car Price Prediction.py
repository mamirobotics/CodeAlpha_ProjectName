# Muhammad Amir Mushtaq
# TASK 3: CAR PRICE PREDICTION
# CodeAlpha Data Science Internship
# First Need to Install the Required Libraries if not already you have them

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


# 1) Load Dataset

df = pd.read_csv(r"C:\Users\Acer\Downloads\Code Alpha Internship\Task Code and Results\Data Scince Internship\car data.csv")  # (name may vary, rename accordingly)

print("\nDataset Shape:", df.shape)
print(df.head())
print("\nColumns:", df.columns)

# 2) Cleaning

df.dropna(inplace=True)

# Standardize column names
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# Check if selling_price exists
target_candidates = ["selling_price", "price"]
target_col = None
for t in target_candidates:
    if t in df.columns:
        target_col = t
        break

if target_col is None:
    raise ValueError("Target column not found! Please check dataset columns.")

# 3) Feature Engineering

# Example: create car age if year exists
if "year" in df.columns:
    df["car_age"] = 2025 - df["year"]

# Remove extremely irrelevant columns if exist
drop_cols = []
for col in ["name", "torque"]:
    if col in df.columns:
        drop_cols.append(col)

df.drop(columns=drop_cols, inplace=True, errors="ignore")

X = df.drop(target_col, axis=1)
y = df[target_col]

# 4) Identify categorical/numerical

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()


# 5) Preprocessing

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

# 6) Models

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42)
}

# 7) Train/Test Split


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 8) Train + Evaluate

results = []

for name, model in models.items():
    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results.append((name, mae, rmse, r2))

    print("\nAMIR")
    print("MODEL:", name)
    print("Muhammad Amir Mushtaq")
    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2  :", round(r2, 4))

# 9) Results Table

results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2"]).sort_values(by="R2", ascending=False)
print("\nFinal Results:\n", results_df)

# 10) Plot Actual vs Predicted (Best Model)


best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]

best_pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", best_model)
])

best_pipe.fit(X_train, y_train)
best_preds = best_pipe.predict(X_test)

plt.figure(figsize=(7, 6))
plt.scatter(y_test, best_preds)
plt.title(f"Actual vs Predicted - {best_model_name}")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.grid(True)
plt.tight_layout()
plt.show()
