# Muhammad Amir Mushtaq
# TASK 1: IRIS FLOWER CLASSIFICATION
# CodeAlpha Data Science Internship

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# 1) Load Dataset

df = pd.read_csv(r"C:\Users\Acer\Downloads\Code Alpha Internship\Task Code and Results\Data Scince Internship\Iris.csv")

print("\nDataset Shape:", df.shape)
print(df.head())

# 2) Data Cleaning
if "Id" in df.columns:
    df.drop("Id", axis=1, inplace=True)

print("\nMissing Values:\n", df.isnull().sum())


# 3) Encode Target

le = LabelEncoder()
df["Species"] = le.fit_transform(df["Species"])

# Save class names for later
class_names = le.classes_  # ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
print("\nFlower Classes:", class_names)

X = df.drop("Species", axis=1)
y = df["Species"]


# 4) Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


# 5) Scaling

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 6) We are train  our different models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM (RBF Kernel)": SVC(kernel="rbf"),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

model_results = []
species_results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    # Overall Accuracy
    acc = accuracy_score(y_test, preds)
    model_results.append((name, acc))

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)

    # Species-wise Accuracy
    per_species_accuracy = cm.diagonal() / cm.sum(axis=1)

    species_results[name] = per_species_accuracy

    print("\nAAAAAAAAAAAAAAAAAAAA")
    print(f"MODEL: {name}")
    print("BBBBBBBBBBBBBBBBBBBBBB")
    print("Overall Accuracy:", round(acc, 4))

    print("\nConfusion Matrix:\n", cm)

    print("\nClassification Report:\n",
          classification_report(y_test, preds, target_names=class_names))

    print("\nSpecies-wise Accuracy:")
    for i, flower in enumerate(class_names):
        print(f"{flower}: {round(per_species_accuracy[i], 4)}")


# 7) Model Accuracy Plot

results_df = pd.DataFrame(model_results, columns=["Model", "Accuracy"]).sort_values(by="Accuracy", ascending=False)

plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.xticks(rotation=20)
plt.title("Iris Classification - Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()

print("\nFinal Model Accuracy Results:\n")
print(results_df)

# 8) Best Model Species Comparison Plot

best_model_name = results_df.iloc[0]["Model"]
best_species_acc = species_results[best_model_name]

plt.figure(figsize=(8, 5))
plt.bar(class_names, best_species_acc)
plt.title(f"Species-wise Accuracy (Best Model: {best_model_name})")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()


# 9) Species Accuracy Table (All Models)

species_table = pd.DataFrame(species_results, index=class_names)
species_table = species_table.T  # models as rows
print("\nSpecies-wise Accuracy Table (All Models):\n")
print(species_table)
