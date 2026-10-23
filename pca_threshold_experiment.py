
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "breast_cancer_wisconsin.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
X = df.drop(columns="diagnosis")
y = df["diagnosis"].map({"B": 0, "M": 1})

if y.isna().any():
    raise ValueError("Unexpected diagnosis labels.")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

approaches = {
    "Original features": None,
    "PCA - 80%": 0.80,
    "PCA - 85%": 0.85,
    "PCA - 90%": 0.90,
    "PCA - 95%": 0.95,
    "PCA - 99%": 0.99,
}

rows = []

for name, threshold in approaches.items():
    steps = [("scaler", StandardScaler())]

    if threshold is not None:
        steps.append(
            ("pca", PCA(
                n_components=threshold,
                svd_solver="full",
            ))
        )

    steps.append(
        ("model", LogisticRegression(max_iter=5000))
    )

    pipeline = Pipeline(steps)

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_estimator=True,
        n_jobs=-1,
    )

    if threshold is None:
        component_counts = [X.shape[1]] * cv.n_splits
    else:
        component_counts = [
            estimator.named_steps["pca"].n_components_
            for estimator in scores["estimator"]
        ]

    rows.append({
        "Approach": name,
        "Variance threshold": (
            "N/A" if threshold is None else threshold
        ),
        "Mean components": sum(component_counts) / len(component_counts),
        "Accuracy mean": scores["test_accuracy"].mean(),
        "Accuracy std": scores["test_accuracy"].std(),
        "Balanced accuracy mean":
            scores["test_balanced_accuracy"].mean(),
        "F1 mean": scores["test_f1"].mean(),
        "ROC-AUC mean": scores["test_roc_auc"].mean(),
        "ROC-AUC std": scores["test_roc_auc"].std(),
    })

results = pd.DataFrame(rows)
results.to_csv(
    OUTPUT_DIR / "pca_threshold_results.csv",
    index=False,
)

print("\nPCA THRESHOLD EXPERIMENT")
print(results.round(4).to_string(index=False))
print("\nSaved to outputs/pca_threshold_results.csv")
