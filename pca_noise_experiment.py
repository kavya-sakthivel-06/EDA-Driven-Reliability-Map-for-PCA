
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
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
X = df.drop(columns="diagnosis").to_numpy(dtype=float)
y = df["diagnosis"].map({"B": 0, "M": 1}).to_numpy()

if pd.isna(y).any():
    raise ValueError("Unexpected diagnosis labels.")

NOISE_LEVELS = [0.00, 0.05, 0.10, 0.20]
COMPONENT_COUNTS = [5, 10, 17]
REPEATS = 20
RNG = np.random.default_rng(42)

# Standardize once to define noise in comparable feature units.
# These transformed data are used only for this controlled stress test.
X_standardized = StandardScaler().fit_transform(X)

# Reuse the same random noise pattern at different strengths.
noise_pattern = RNG.normal(size=X_standardized.shape)

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

performance_rows = []
stability_rows = []

for noise_level in NOISE_LEVELS:
    print(f"\nRunning noise level: {noise_level:.2f}")

    X_noisy = (
        X_standardized + noise_level * noise_pattern
    )

    # Part A: predictive performance.
    approaches = {
        "Original features": None,
        "PCA - 95% variance": 0.95,
    }

    for approach, threshold in approaches.items():
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
            X_noisy,
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

        performance_rows.append({
            "Noise level": noise_level,
            "Approach": approach,
            "Mean components": np.mean(component_counts),
            "Accuracy mean": scores["test_accuracy"].mean(),
            "Accuracy std": scores["test_accuracy"].std(),
            "Balanced accuracy mean":
                scores["test_balanced_accuracy"].mean(),
            "F1 mean": scores["test_f1"].mean(),
            "ROC-AUC mean": scores["test_roc_auc"].mean(),
            "ROC-AUC std": scores["test_roc_auc"].std(),
        })

    # Part B: stability across repeated random subsets.
    sample_size = int(0.80 * len(X_noisy))

    subsets = [
        RNG.choice(len(X_noisy), size=sample_size, replace=False)
        for _ in range(REPEATS)
    ]

    for k in COMPONENT_COUNTS:
        fitted_components = []

        for indices in subsets:
            subset = X_noisy[indices]
            subset_scaled = StandardScaler().fit_transform(subset)

            pca = PCA(n_components=k, svd_solver="full")
            pca.fit(subset_scaled)
            fitted_components.append(pca.components_)

        similarities = []

        for i, j in combinations(range(REPEATS), 2):
            angles = subspace_angles(
                fitted_components[i].T,
                fitted_components[j].T,
            )
            similarities.append(
                np.mean(np.cos(angles) ** 2)
            )

        stability_rows.append({
            "Noise level": noise_level,
            "Sample size": sample_size,
            "Components": k,
            "Mean subspace similarity": np.mean(similarities),
            "Similarity std": np.std(similarities),
            "Pairs compared": len(similarities),
        })

performance = pd.DataFrame(performance_rows)
stability = pd.DataFrame(stability_rows)

performance_path = OUTPUT_DIR / "pca_noise_performance.csv"
stability_path = OUTPUT_DIR / "pca_noise_stability.csv"

performance.to_csv(performance_path, index=False)
stability.to_csv(stability_path, index=False)

print("\nNOISE EXPERIMENT: CLASSIFICATION PERFORMANCE")
print(performance.round(4).to_string(index=False))

print("\nNOISE EXPERIMENT: SUBSPACE STABILITY")
print(stability.round(4).to_string(index=False))

print("\nSaved results:")
print(performance_path)
print(stability_path)
