
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize


DATA_PATH = Path("data/wine.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df.drop(columns="target")
y = df["target"]

thresholds = [None, 0.80, 0.85, 0.90, 0.95, 0.99]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

classes = np.sort(y.unique())
results = []

for threshold in thresholds:
    fold_metrics = []
    component_counts = []

    for train_idx, test_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        steps = [("scaler", StandardScaler())]

        if threshold is not None:
            steps.append(
                ("pca", PCA(n_components=threshold))
            )

        steps.append(
            ("model", LogisticRegression(max_iter=5000))
        )

        model = Pipeline(steps)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        y_test_binary = label_binarize(
            y_test, classes=classes
        )

        fold_metrics.append({
            "accuracy": accuracy_score(y_test, predictions),
            "balanced_accuracy": balanced_accuracy_score(
                y_test, predictions
            ),
            "macro_f1": f1_score(
                y_test, predictions, average="macro"
            ),
            "roc_auc_ovr": roc_auc_score(
                y_test_binary,
                probabilities,
                multi_class="ovr",
                average="macro",
            ),
        })

        if threshold is None:
            component_counts.append(X.shape[1])
        else:
            component_counts.append(
                model.named_steps["pca"].n_components_
            )

    metrics_df = pd.DataFrame(fold_metrics)

    results.append({
        "approach": (
            "Original features"
            if threshold is None
            else f"PCA - {int(threshold * 100)}% variance"
        ),
        "threshold": (
            "None" if threshold is None else threshold
        ),
        "mean_components": np.mean(component_counts),
        "accuracy_mean": metrics_df["accuracy"].mean(),
        "accuracy_std": metrics_df["accuracy"].std(ddof=1),
        "balanced_accuracy_mean":
            metrics_df["balanced_accuracy"].mean(),
        "macro_f1_mean": metrics_df["macro_f1"].mean(),
        "roc_auc_ovr_mean": metrics_df["roc_auc_ovr"].mean(),
        "roc_auc_ovr_std":
            metrics_df["roc_auc_ovr"].std(ddof=1),
    })

results_df = pd.DataFrame(results)
results_df.to_csv(
    OUTPUT_DIR / "wine_baseline_results.csv", index=False
)

print("\nWINE PCA BASELINE RESULTS")
print("=" * 85)
print(results_df.round(4).to_string(index=False))
print("\nSaved to outputs/wine_baseline_results.csv")
