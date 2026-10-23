
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


# -----------------------------------------
# 1. Set up project folders
# -----------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# -----------------------------------------
# 2. Download the official UCI dataset
# -----------------------------------------

print("Downloading Breast Cancer Wisconsin (Diagnostic)...")

uci_dataset = fetch_ucirepo(id=17)

X = uci_dataset.data.features.copy()
y = uci_dataset.data.targets.copy()

# Clean up column names
X.columns = X.columns.astype(str).str.strip()
y.columns = y.columns.astype(str).str.strip()

# Combine features and target
target_name = y.columns[0]

df = X.copy()
df["diagnosis"] = y[target_name].astype(str).str.strip().to_numpy()


# -----------------------------------------
# 3. Save a local copy of the dataset
# -----------------------------------------

dataset_path = DATA_DIR / "breast_cancer_wisconsin.csv"
df.to_csv(dataset_path, index=False)


# -----------------------------------------
# 4. Inspect the dataset structure
# -----------------------------------------

print("\n" + "=" * 55)
print("DATASET OVERVIEW")
print("=" * 55)

print(f"Rows: {df.shape[0]}")
print(f"Feature columns: {X.shape[1]}")
print(f"Target column: diagnosis")

print("\nFirst five rows:")
print(df.head().to_string(index=False))

print("\nData types:")
print(df.dtypes.to_string())


# -----------------------------------------
# 5. Check data quality
# -----------------------------------------

print("\n" + "=" * 55)
print("DATA QUALITY")
print("=" * 55)

missing_values = df.isnull().sum()

print("\nMissing values by column:")
print(missing_values[missing_values > 0].to_string()
      if missing_values.any() else "No missing values found.")

print(f"\nDuplicate rows: {df.duplicated().sum()}")

print("\nDiagnosis counts:")
print(df["diagnosis"].value_counts().to_string())

print("\nDiagnosis percentages:")
print(
    (df["diagnosis"].value_counts(normalize=True) * 100)
    .round(2)
    .to_string()
)


# -----------------------------------------
# 6. Analyse feature correlations
# -----------------------------------------

correlation_matrix = X.corr(numeric_only=True)

correlation_path = OUTPUT_DIR / "feature_correlations.csv"
correlation_matrix.to_csv(correlation_path)

# Extract each unique feature pair once
upper_triangle = correlation_matrix.where(
    pd.DataFrame(
        [
            [i < j for j in range(len(X.columns))]
            for i in range(len(X.columns))
        ],
        index=X.columns,
        columns=X.columns,
    )
)

correlation_pairs = (
    upper_triangle.stack()
    .rename("correlation")
    .reset_index()
)

correlation_pairs.columns = [
    "feature_1",
    "feature_2",
    "correlation",
]

# Sort by absolute correlation, strongest first
correlation_pairs["absolute_correlation"] = (
    correlation_pairs["correlation"].abs()
)

correlation_pairs = correlation_pairs.sort_values(
    "absolute_correlation",
    ascending=False,
)

high_correlations = correlation_pairs[
    correlation_pairs["absolute_correlation"] >= 0.90
].drop(columns="absolute_correlation")

print("\n" + "=" * 55)
print("HIGHLY CORRELATED FEATURE PAIRS")
print("=" * 55)

if high_correlations.empty:
    print("No feature pairs have absolute correlation >= 0.90.")
else:
    print(high_correlations.head(20).to_string(index=False))

high_correlations.to_csv(
    OUTPUT_DIR / "high_correlations.csv",
    index=False,
)


# -----------------------------------------
# 7. Save a concise inspection report
# -----------------------------------------

report_path = OUTPUT_DIR / "dataset_inspection.txt"

with open(report_path, "w", encoding="utf-8") as report:
    report.write("BREAST CANCER WISCONSIN (DIAGNOSTIC)\n")
    report.write("Source: UCI Machine Learning Repository\n")
    report.write("Dataset ID: 17\n\n")

    report.write(f"Observations: {df.shape[0]}\n")
    report.write(f"Features: {X.shape[1]}\n")
    report.write(f"Duplicate rows: {df.duplicated().sum()}\n\n")

    report.write("Missing values:\n")
    report.write(missing_values.to_string())
    report.write("\n\nDiagnosis counts:\n")
    report.write(df["diagnosis"].value_counts().to_string())
    report.write("\n\nHighly correlated feature pairs (|r| >= 0.90):\n")
    report.write(high_correlations.to_string(index=False))

print("\n" + "=" * 55)
print("INSPECTION COMPLETE")
print("=" * 55)

print(f"Dataset saved: {dataset_path}")
print(f"Correlations saved: {correlation_path}")
print(f"High correlations saved: {OUTPUT_DIR / 'high_correlations.csv'}")
print(f"Report saved: {report_path}")
