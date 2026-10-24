
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/wine.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
X = df.drop(columns="target").to_numpy(dtype=float)

sample_fractions = [0.60, 0.75, 0.90]
component_counts = [2,3, 5, 8]
n_subsets = 20
rng = np.random.default_rng(42)

results = []

for fraction in sample_fractions:
    sample_size = int(round(len(X) * fraction))

    # Generate reproducible random subsets
    subsets = [
        rng.choice(len(X), size=sample_size, replace=False)
        for _ in range(n_subsets)
    ]

    for k in component_counts:
        fitted_components = []

        for indices in subsets:
            X_subset = X[indices]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_subset)

            pca = PCA(n_components=k)
            pca.fit(X_scaled)

            fitted_components.append(pca.components_)

        # Compare every pair of fitted PCA subspaces
        similarities = []

        for i, j in combinations(range(n_subsets), 2):
            components_a = fitted_components[i]
            components_b = fitted_components[j]

            # Singular values are cosines of principal angles
            singular_values = np.linalg.svd(
                components_a @ components_b.T,
                compute_uv=False,
            )

            similarity = np.mean(singular_values ** 2)
            similarities.append(similarity)

        results.append({
            "sample_fraction": fraction,
            "sample_size": sample_size,
            "components": k,
            "mean_subspace_similarity": np.mean(similarities),
            "similarity_std": np.std(similarities, ddof=1),
            "pairwise_comparisons": len(similarities),
        })

results_df = pd.DataFrame(results)
results_df.to_csv(
    OUTPUT_DIR / "wine_stability_results.csv",
    index=False,
)

print("\nWINE PCA SUBSPACE STABILITY")
print("=" * 85)
print(results_df.round(4).to_string(index=False))
print("\nSaved to outputs/wine_stability_results.csv")
