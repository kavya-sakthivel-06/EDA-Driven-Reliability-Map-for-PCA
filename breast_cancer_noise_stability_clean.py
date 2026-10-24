
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = Path("data/breast_cancer_wisconsin.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLE_FRACTIONS = [0.60, 0.90]
COMPONENT_COUNTS = [2, 3, 5, 8, 10]
NOISE_LEVELS = [0.00, 0.05, 0.10, 0.20]
N_REPEATS = 20
RANDOM_SEED = 42


def subspace_similarity(components_a, components_b):
    """
    Mean squared cosine of the principal angles.
    1.0 = identical subspaces.
    Values closer to 0 = less similar subspaces.
    """
    singular_values = np.linalg.svd(
        components_a @ components_b.T,
        compute_uv=False
    )
    return np.mean(np.clip(singular_values**2, 0, 1))


# -----------------------------
# Load real dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["diagnosis"]).select_dtypes(
    include=[np.number]
).to_numpy(dtype=float)

print("Dataset shape:", X.shape)

rng = np.random.default_rng(RANDOM_SEED)
results = []

# -----------------------------
# Stability under noise
# -----------------------------
for fraction in SAMPLE_FRACTIONS:
    sample_size = int(len(X) * fraction)

    for repeat in range(N_REPEATS):
        indices = rng.choice(
            len(X), size=sample_size, replace=False
        )
        sample = X[indices]

        # Fit scaling only on this sampled subset
        scaler = StandardScaler()
        clean_scaled = scaler.fit_transform(sample)

        # Use one common standardized clean sample as reference
        max_k = max(COMPONENT_COUNTS)
        clean_pca = PCA(n_components=max_k)
        clean_pca.fit(clean_scaled)

        for noise in NOISE_LEVELS:
            # Same clean sample; perturbations vary by noise level
            noise_rng = np.random.default_rng(
                RANDOM_SEED + repeat * 1000
                + int(fraction * 100) * 10
                + int(noise * 100)
            )

            noisy_scaled = clean_scaled + noise_rng.normal(
                loc=0.0,
                scale=noise,
                size=clean_scaled.shape
            )

            noisy_pca = PCA(n_components=max_k)
            noisy_pca.fit(noisy_scaled)

            for k in COMPONENT_COUNTS:
                similarity = subspace_similarity(
                    clean_pca.components_[:k],
                    noisy_pca.components_[:k]
                )

                results.append({
                    "sample_fraction": fraction,
                    "sample_size": sample_size,
                    "repeat": repeat + 1,
                    "noise_level": noise,
                    "n_components": k,
                    "subspace_similarity": similarity
                })

# -----------------------------
# Save detailed and summary data
# -----------------------------
detail = pd.DataFrame(results)

summary = (
    detail.groupby(
        ["sample_fraction", "sample_size",
         "noise_level", "n_components"]
    )["subspace_similarity"]
    .agg(["mean", "std"])
    .reset_index()
)

detail.to_csv(
    OUTPUT_DIR / "breast_cancer_noise_stability_clean_detail.csv",
    index=False
)
summary.to_csv(
    OUTPUT_DIR / "breast_cancer_noise_stability_clean_summary.csv",
    index=False
)

print("\nBREAST CANCER PCA SUBSPACE STABILITY UNDER NOISE")
print("=" * 80)
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nSaved detailed and summary results in outputs/")
