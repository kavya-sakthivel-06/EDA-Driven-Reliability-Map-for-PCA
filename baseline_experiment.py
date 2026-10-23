
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. Reproducibility and output folders
# --------------------------------------------------

RANDOM_SEED = 42

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Generate a synthetic classification dataset
# --------------------------------------------------

X, y = make_classification(
    n_samples=2000,
    n_features=30,
    n_informative=8,
    n_redundant=12,
    n_repeated=0,
    n_classes=2,
    weights=[0.65, 0.35],
    class_sep=1.0,
    flip_y=0.02,
    random_state=RANDOM_SEED,
)

feature_names = [f"feature_{i + 1}" for i in range(X.shape[1])]

dataset = pd.DataFrame(X, columns=feature_names)
dataset["target"] = y

dataset.to_csv(DATA_DIR / "synthetic_baseline.csv", index=False)

print("DATASET INFORMATION")
print("-------------------")
print(f"Number of rows: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of informative features: 8")
print(f"Number of redundant features: 12")
print(f"Class proportions:\n{pd.Series(y).value_counts(normalize=True).sort_index()}")


# --------------------------------------------------
# 3. Split data before fitting any preprocessing
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_SEED,
)


# --------------------------------------------------
# 4. Define two comparable machine-learning pipelines
# --------------------------------------------------

original_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)),
])

pca_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95, svd_solver="full")),
    ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)),
])


# --------------------------------------------------
# 5. Train and evaluate both pipelines
# --------------------------------------------------

results = []

experiments = [
    ("Original features", original_pipeline),
    ("PCA - 95% variance", pca_pipeline),
]

for experiment_name, pipeline in experiments:
    start_time = time.perf_counter()

    pipeline.fit(X_train, y_train)

    elapsed_seconds = time.perf_counter() - start_time

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    row = {
        "Experiment": experiment_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Balanced Accuracy": balanced_accuracy_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "Training Time (seconds)": elapsed_seconds,
    }

    if experiment_name == "PCA - 95% variance":
        fitted_pca = pipeline.named_steps["pca"]
        row["Components Retained"] = fitted_pca.n_components_
        row["Actual Variance Retained"] = (
            fitted_pca.explained_variance_ratio_.sum()
        )
    else:
        row["Components Retained"] = X.shape[1]
        row["Actual Variance Retained"] = None

    results.append(row)


# --------------------------------------------------
# 6. Save and display results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_path = OUTPUT_DIR / "baseline_results.csv"
results_df.to_csv(results_path, index=False)

print("\nBASELINE RESULTS")
print("----------------")
print(results_df.round(4).to_string(index=False))

original_auc = results_df.loc[
    results_df["Experiment"] == "Original features", "ROC-AUC"
].iloc[0]

pca_auc = results_df.loc[
    results_df["Experiment"] == "PCA - 95% variance", "ROC-AUC"
].iloc[0]

print("\nINITIAL FINDING")
print("---------------")
print(f"ROC-AUC without PCA: {original_auc:.4f}")
print(f"ROC-AUC with PCA:    {pca_auc:.4f}")
print(f"ROC-AUC change:      {pca_auc - original_auc:+.4f}")

print(f"\nResults saved to: {results_path}")


# --------------------------------------------------
# 7. Create a baseline comparison plot
# --------------------------------------------------

metrics = ["Accuracy", "F1 Score", "ROC-AUC"]

plot_df = results_df.set_index("Experiment")[metrics]

ax = plot_df.plot(kind="bar", figsize=(9, 5), ylim=(0, 1))
ax.set_title("Baseline: Original Features vs PCA")
ax.set_ylabel("Score")
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=0)
ax.legend(loc="lower right")
plt.tight_layout()

plot_path = OUTPUT_DIR / "baseline_comparison.png"
plt.savefig(plot_path, dpi=200)
plt.close()

print(f"Plot saved to: {plot_path}")
print("\nBaseline experiment completed successfully.")
