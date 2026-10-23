
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "breast_cancer_wisconsin.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
X = df.drop(columns="diagnosis").to_numpy(dtype=float)

SAMPLE_FRACTIONS = [0.70, 0.80, 0.90]
COMPONENT_COUNTS = [5, 10, 17]
REPEATS = 20
RNG = np.random.default_rng(42)

rows = []

for fraction in SAMPLE_FRACTIONS:
    sample_size = int(len(X) * fraction)
    print(f"\nSample fraction: {fraction:.0%} ({sample_size} rows)")

    # Draw random subsets without replacement.
    subsets = [
        RNG.choice(len(X), size=sample_size, replace=False)
        for _ in range(REPEATS)
    ]

    for k in COMPONENT_COUNTS:
        fitted_components = []

        for indices in subsets:
            X_subset = X[indices]

            # Fit scaling independently on each subset.
            X_scaled = StandardScaler().fit_transform(X_subset)

            pca = PCA(n_components=k, svd_solver="full")
            pca.fit(X_scaled)

            fitted_components.append(pca.components_)

        similarities = []

        # Compare all pairs of fitted subspaces.
        for i, j in combinations(range(REPEATS), 2):
            angles = subspace_angles(
                fitted_components[i].T,
                fitted_components[j].T,
            )

            similarity = np.mean(np.cos(angles) ** 2)
            similarities.append(similarity)

        rows.append({
            "Sample fraction": fraction,
            "Sample size": sample_size,
            "Components": k,
            "Mean subspace similarity": np.mean(similarities),
            "Similarity std": np.std(similarities),
            "Pairs compared": len(similarities),
        })

results = pd.DataFrame(rows)

output_path = OUTPUT_DIR / "pca_stability_results.csv"
results.to_csv(output_path, index=False)

print("\nPCA SUBSPACE STABILITY RESULTS")
print(results.round(4).to_string(index=False))
print("\nResults saved to:", output_path)
