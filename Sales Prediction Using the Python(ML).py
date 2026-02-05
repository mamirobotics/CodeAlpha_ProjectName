# Muhammad Amir Mushtaq
# Optional Task of Internship
# TASK 4: SALES PREDICTION
# CodeAlpha Data Science Internship


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1) Load Dataset

df = pd.read_csv(r"C:\Users\Acer\Downloads\Code Alpha Internship\Task Code and Results\Data Scince Internship\Advertising.csv")

print("\nDataset Shape:", df.shape)
print(df.head())
print(df.columns)

# 2) Clean
# 
df.dropna(inplace=True)

# Some versions have "Unnamed: 0"
for col in df.columns:
    if "Unnamed" in col:
        df.drop(col, axis=1, inplace=True)

# 3) EDA - Correlation
print("\nCorrelation:\n", df.corr(numeric_only=True))


# 4) Visualize Advertising Impact

plt.figure(figsize=(7, 5))
plt.scatter(df["TV"], df["Sales"])
plt.title("TV Spend vs Sales")
plt.xlabel("TV Advertising Spend")
plt.ylabel("Sales")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(df["Radio"], df["Sales"])
plt.title("Radio Spend vs Sales")
plt.xlabel("Radio Advertising Spend")
plt.ylabel("Sales")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(df["Newspaper"], df["Sales"])
plt.title("Newspaper Spend vs Sales")
plt.xlabel("Newspaper Advertising Spend")
plt.ylabel("Sales")
plt.grid(True)
plt.tight_layout()
plt.show()

# 5) Features/Target

X = df.drop("Sales", axis=1)
y = df["Sales"]

# 6) Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)


# 7) Train Models

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results.append((name, mae, rmse, r2))

    print("\nPakistan")
    print("MODEL:", name)
    print("India ")
    print("MAE :", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("R2  :", round(r2, 4))

# 8) Compare Models

# Why we compare these models you know it but it not hit and trail method to reach the optimization.
results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2"]).sort_values(by="R2", ascending=False)
print("\nFinal Results:\n", results_df)

# 9) Plot Actual vs Predicted (Best)
best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]

best_model.fit(X_train, y_train)
best_preds = best_model.predict(X_test)

plt.figure(figsize=(7, 6))
plt.scatter(y_test, best_preds)
plt.title(f"Actual vs Predicted - {best_model_name}")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.grid(True)
plt.tight_layout()
plt.show()
