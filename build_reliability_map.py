
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# These are the latest experiments with fold-specific preprocessing.
FILES = {
    "Breast Cancer — performance": (
        "breast_cancer_noise_performance_clean_summary.csv"
    ),
    "Breast Cancer — stability": (
        "breast_cancer_noise_stability_clean_summary.csv"
    ),
    "Wine — performance": (
        "wine_noise_performance_clean_summary.csv"
    ),
    "Wine — stability": (
        "wine_noise_stability.csv"
    ),
}

# Load and inspect the files before plotting.
data = {}

for name, filename in FILES.items():
    path = OUTPUT_DIR / filename

    if path.exists():
        df = pd.read_csv(path)
        data[name] = df
        print(f"\n{name}: {filename}")
        print("Columns:", df.columns.tolist())
        print(df.head(3).to_string(index=False))
    else:
        print(f"\nMISSING: {path}")

if len(data) != len(FILES):
    print(
        "\nSome expected files are missing. "
        "Check the filenames above before continuing."
    )
    raise SystemExit(1)

# -------------------------------------------------
# 1. Plot predictive accuracy under noise
# -------------------------------------------------
performance_files = [
    ("Breast Cancer — performance",
     "breast_cancer_accuracy_under_noise.png"),
    ("Wine — performance",
     "wine_accuracy_under_noise.png"),
]

for dataset_name, plot_name in performance_files:
    df = data[dataset_name].copy()

    # Normalize common column-name variations.
    noise_col = next(
        (c for c in df.columns if c.lower() in
         ["noise", "noise_level"]),
        None
    )
    approach_col = next(
        (c for c in df.columns if c.lower() in
         ["approach", "method", "model"]),
        None
    )
    accuracy_col = next(
        (c for c in df.columns if c.lower() in
         ["accuracy_mean", "accuracy", "mean_accuracy"]),
        None
    )

    if not all([noise_col, approach_col, accuracy_col]):
        print(
            f"Skipping {dataset_name}: "
            "could not identify noise, approach, and accuracy columns."
        )
        continue

    fig, ax = plt.subplots(figsize=(8, 5))

    for approach, group in df.groupby(approach_col):
        group = group.sort_values(noise_col)
        ax.plot(
            group[noise_col],
            group[accuracy_col],
            marker="o",
            label=str(approach)
        )

    ax.set(
        title=f"{dataset_name}: Accuracy Under Noise",
        xlabel="Gaussian noise level (standardized feature units)",
        ylabel="Mean cross-validation accuracy"
    )
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / plot_name, dpi=300)
    plt.close(fig)
    print("Saved:", OUTPUT_DIR / plot_name)

# -------------------------------------------------
# 2. Plot subspace stability under noise
# -------------------------------------------------

# -------------------------------------------------
# 2. Plot subspace stability under noise
# -------------------------------------------------
stability_files = [
    (
        "Breast Cancer — stability",
        "breast_cancer_stability_loss.png",
        "noise_level",
        "n_components",
        "mean",
        "std",
    ),
    (
        "Wine — stability",
        "wine_stability_loss.png",
        "noise",
        "components",
        "mean_subspace_similarity",
        "similarity_std",
    ),
]

for (
    dataset_name,
    plot_name,
    noise_col,
    k_col,
    similarity_col,
    std_col,
) in stability_files:

    df = data[dataset_name].copy()

    required = [noise_col, k_col, similarity_col]
    if not all(col in df.columns for col in required):
        print(f"Skipping {dataset_name}: required columns missing.")
        continue

    df["stability_loss"] = 1 - df[similarity_col]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Separate lines for each sample fraction.
    for fraction, fraction_group in df.groupby("sample_fraction"):
        for k, group in fraction_group.groupby(k_col):
            group = group.sort_values(noise_col)

            ax.plot(
                group[noise_col],
                group["stability_loss"],
                marker="o",
                label=f"{int(k)} components, {fraction:.0%} sample"
            )

    ax.set(
        title=f"{dataset_name}: PCA Subspace Stability Loss",
        xlabel="Gaussian noise level",
        ylabel="Stability loss (1 − subspace similarity)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / plot_name, dpi=300)
    plt.close(fig)

    print("Saved:", OUTPUT_DIR / plot_name)
