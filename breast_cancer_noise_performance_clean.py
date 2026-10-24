
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
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/breast_cancer_wisconsin.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df.drop(columns="diagnosis").to_numpy(dtype=float)
y = df["diagnosis"].map({"B": 0, "M": 1}).to_numpy()

noise_levels = [0.00, 0.05, 0.10, 0.20]
approaches = ["Original features", "PCA - 95% variance"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rng = np.random.default_rng(42)

results = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Fit scaling on training data only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Use matched perturbations for both model approaches
    train_noise_pattern = rng.normal(size=X_train.shape)
    test_noise_pattern = rng.normal(size=X_test.shape)

    for noise in noise_levels:
        X_train_noisy = X_train + noise * train_noise_pattern
        X_test_noisy = X_test + noise * test_noise_pattern

        for approach in approaches:
            if approach == "PCA - 95% variance":
                pca = PCA(n_components=0.95)
                X_train_model = pca.fit_transform(X_train_noisy)
                X_test_model = pca.transform(X_test_noisy)
                n_components = pca.n_components_
            else:
                X_train_model = X_train_noisy
                X_test_model = X_test_noisy
                n_components = X.shape[1]

            model = LogisticRegression(max_iter=5000)
            model.fit(X_train_model, y_train)

            predictions = model.predict(X_test_model)
            probabilities = model.predict_proba(X_test_model)[:, 1]

            results.append({
                "fold": fold,
                "noise": noise,
                "approach": approach,
                "components": n_components,
                "accuracy": accuracy_score(y_test, predictions),
                "balanced_accuracy": balanced_accuracy_score(
                    y_test, predictions
                ),
                "macro_f1": f1_score(
                    y_test, predictions, average="macro"
                ),
                "roc_auc": roc_auc_score(y_test, probabilities),
            })

results_df = pd.DataFrame(results)

summary = (
    results_df
    .groupby(["noise", "approach"])
    .agg(
        mean_components=("components", "mean"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        macro_f1_mean=("macro_f1", "mean"),
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
    )
    .reset_index()
)

results_df.to_csv(
    OUTPUT_DIR / "breast_cancer_noise_performance_clean_folds.csv",
    index=False,
)
summary.to_csv(
    OUTPUT_DIR / "breast_cancer_noise_performance_clean_summary.csv",
    index=False,
)

print("\nBREAST CANCER PCA PERFORMANCE UNDER NOISE")
print("=" * 100)
print(summary.round(4).to_string(index=False))
print("\nSaved fold-level and summary results in outputs/")
