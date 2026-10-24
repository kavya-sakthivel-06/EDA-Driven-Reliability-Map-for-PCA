
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/wine.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
X = df.drop(columns="target").to_numpy(dtype=float)

noise_levels = [0.00, 0.05, 0.10, 0.20]
sample_fractions = [0.60, 0.90]
component_counts = [2, 3, 5, 8]
n_subsets = 20

rng = np.random.default_rng(42)
results = []

for fraction in sample_fractions:
    sample_size = int(round(len(X) * fraction))

    for subset_id in range(n_subsets):
        indices = rng.choice(
            len(X), size=sample_size, replace=False
        )

        # Scale each subset independently.
        X_subset = X[indices]
        X_scaled = StandardScaler().fit_transform(X_subset)

        for k in component_counts:
            clean_pca = PCA(n_components=k)
            clean_pca.fit(X_scaled)
            clean_components = clean_pca.components_

            for noise in noise_levels:
                perturbed = X_scaled + rng.normal(
                    loc=0.0,
                    scale=noise,
                    size=X_scaled.shape,
                )

                noisy_pca = PCA(n_components=k)
                noisy_pca.fit(perturbed)

                # Principal-angle subspace similarity.
                singular_values = np.linalg.svd(
                    clean_components @ noisy_pca.components_.T,
                    compute_uv=False,
                )

                similarity = np.mean(singular_values ** 2)

                results.append({
                    "sample_fraction": fraction,
                    "sample_size": sample_size,
                    "subset_id": subset_id,
                    "components": k,
                    "noise": noise,
                    "subspace_similarity": similarity,
                })

results_df = pd.DataFrame(results)

summary = (
    results_df
    .groupby([
        "sample_fraction",
        "sample_size",
        "components",
        "noise",
    ])["subspace_similarity"]
    .agg(["mean", "std"])
    .reset_index()
    .rename(columns={
        "mean": "mean_subspace_similarity",
        "std": "similarity_std",
    })
)

summary.to_csv(
    OUTPUT_DIR / "wine_noise_stability.csv", index=False
)

print("\nWINE PCA STABILITY UNDER NOISE")
print("=" * 85)
print(summary.round(4).to_string(index=False))
print("\nSaved to outputs/wine_noise_stability.csv")
