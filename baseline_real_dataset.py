
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

print("Dataset shape:", X.shape)
print("\nClass distribution:")
print(y.value_counts().rename(index={0: "Benign", 1: "Malignant"}))

models = {
    "Original features": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000)),
    ]),
    "PCA - 95% variance": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full")),
        ("model", LogisticRegression(max_iter=5000)),
    ]),
}

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

rows = []

for name, pipeline in models.items():
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_estimator=True,
        n_jobs=-1,
    )

    component_counts = (
        [est.named_steps["pca"].n_components_
         for est in scores["estimator"]]
        if name == "PCA - 95% variance"
        else [X.shape[1]] * cv.n_splits
    )

    rows.append({
        "Approach": name,
        "Accuracy mean": scores["test_accuracy"].mean(),
        "Accuracy std": scores["test_accuracy"].std(),
        "Balanced accuracy mean":
            scores["test_balanced_accuracy"].mean(),
        "F1 mean": scores["test_f1"].mean(),
        "ROC-AUC mean": scores["test_roc_auc"].mean(),
        "ROC-AUC std": scores["test_roc_auc"].std(),
        "Components used": round(
            sum(component_counts) / len(component_counts), 1
        ),
    })

results = pd.DataFrame(rows)
output_path = OUTPUT_DIR / "real_dataset_baseline.csv"
results.to_csv(output_path, index=False)

print("\nFIVE-FOLD CROSS-VALIDATION RESULTS")
print(results.round(4).to_string(index=False))

auc_original = results.loc[
    results["Approach"] == "Original features", "ROC-AUC mean"
].iloc[0]

auc_pca = results.loc[
    results["Approach"] == "PCA - 95% variance", "ROC-AUC mean"
].iloc[0]

print(f"\nROC-AUC change (PCA - original): {auc_pca - auc_original:+.4f}")
print("Results saved to:", output_path)
