
from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path

# Fetch the real Wine dataset from the UCI repository
wine = fetch_ucirepo(id=109)

X = wine.data.features
y = wine.data.targets

print("DATASET OVERVIEW")
print("-" * 40)
print("Rows:", X.shape[0])
print("Features:", X.shape[1])
print("Target columns:", list(y.columns))
print("\nFeature names:")
print(list(X.columns))

print("\nMissing values:", X.isnull().sum().sum())
print("Duplicate feature rows:", X.duplicated().sum())

target = y.iloc[:, 0]
print("\nClass distribution:")
print(target.value_counts().sort_index())

print("\nFeature summary:")
print(X.describe().round(2))

# Save a local copy; data/ is excluded from Git
output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

dataset = X.copy()
dataset["target"] = target
dataset.to_csv(output_dir / "wine.csv", index=False)

print("\nSaved dataset to data/wine.csv")
