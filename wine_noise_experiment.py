
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
X = df.drop(columns="target").to_numpy(dtype=float)
y = df["target"].to_numpy()
classes = np.sort(np.unique(y))

noise_levels = [0.00, 0.05, 0.10, 0.20]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Standardize once to define the controlled perturbation scale.
# Each noise level uses the same random perturbation pattern.
X_standardized = StandardScaler().fit_transform(X)
rng = np.random.default_rng(42)
noise_pattern = rng.normal(size=X_standardized.shape)

performance_results = []

for noise in noise_levels:
    X_noisy = X_standardized + noise * noise_pattern

    for approach in ["Original features", "PCA - 95% variance"]:
        fold_results = []
        component_counts = []

        for train_idx, test_idx in cv.split(X_noisy, y):
            X_train, X_test = X_noisy[train_idx], X_noisy[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            steps = [("scaler", StandardScaler())]

            if approach == "PCA - 95% variance":
                steps.append(("pca", PCA(n_components=0.95)))

            steps.append(
                ("model", LogisticRegression(max_iter=5000))
            )

            model = Pipeline(steps)
            model.fit(X_train, y_train)

            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)

            fold_results.append({
                "accuracy": accuracy_score(y_test, pred),
                "balanced_accuracy": balanced_accuracy_score(
                    y_test, pred
                ),
                "macro_f1": f1_score(
                    y_test, pred, average="macro"
                ),
                "roc_auc_ovr": roc_auc_score(
                    label_binarize(y_test, classes=classes),
                    prob,
                    average="macro",
                ),
            })

            if approach == "PCA - 95% variance":
                component_counts.append(
                    model.named_steps["pca"].n_components_
                )
            else:
                component_counts.append(X.shape[1])

        metrics = pd.DataFrame(fold_results)

        performance_results.append({
            "noise": noise,
            "approach": approach,
            "mean_components": np.mean(component_counts),
            "accuracy_mean": metrics["accuracy"].mean(),
            "accuracy_std": metrics["accuracy"].std(ddof=1),
            "balanced_accuracy_mean":
                metrics["balanced_accuracy"].mean(),
            "macro_f1_mean": metrics["macro_f1"].mean(),
            "roc_auc_ovr_mean": metrics["roc_auc_ovr"].mean(),
        })

performance_df = pd.DataFrame(performance_results)
performance_df.to_csv(
    OUTPUT_DIR / "wine_noise_performance.csv", index=False
)

print("\nWINE PCA PERFORMANCE UNDER NOISE")
print("=" * 90)
print(performance_df.round(4).to_string(index=False))
print("\nSaved to outputs/wine_noise_performance.csv")
