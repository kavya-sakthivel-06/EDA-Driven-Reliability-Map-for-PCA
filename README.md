
# Beyond Explained Variance: An EDA-Driven Reliability Map for PCA

An empirical study of Principal Component Analysis (PCA) under controlled noise, limited sample sizes, and dimensionality reduction, using two real-world datasets.

## 1. Project Overview

Principal Component Analysis (PCA) is widely used for dimensionality reduction. However, retaining a high percentage of explained variance does not automatically guarantee better predictive performance or stable principal components.

This project investigates PCA from multiple perspectives:

- **Compression:** How many components are required to retain a chosen percentage of variance?
- **Predictive performance:** How does PCA compare with the original features?
- **Noise sensitivity:** How does controlled Gaussian noise affect predictive performance and subspace stability?
- **Sample-size sensitivity:** How does the amount of available data affect PCA subspace stability?

The goal is to develop an exploratory, multi-dimensional assessment of PCA reliability rather than relying on explained variance alone.

## 2. Research Questions

1. How does PCA compression affect classification performance compared with using the original features?
2. How does PCA subspace similarity change as controlled perturbation noise increases?
3. How do sample fraction and the number of retained components relate to subspace stability?
4. Can predictive performance, compression, and stability together provide a more informative assessment of PCA?

## 3. Datasets

### Dataset A: Breast Cancer Wisconsin (Diagnostic)

- Observations: 569
- Numerical predictors: 30
- Task: Binary classification
- Source: UCI Machine Learning Repository

Citation:

Wolberg, W., Mangasarian, O., Street, N., & Street, N. (1993). *Breast Cancer Wisconsin (Diagnostic)*. UCI Machine Learning Repository.

https://doi.org/10.24432/C5DW2B

### Dataset B: Wine Recognition

- Observations: 178
- Numerical predictors: 13
- Task: Three-class classification
- Source: UCI Machine Learning Repository

Citation:

Aeberhard, S., Coomans, D., & de Vel (1992). *Wine recognition data*. UCI Machine Learning Repository.

https://doi.org/10.24432/C5PC7J

The datasets differ in sample size, dimensionality, and classification difficulty. Their results are treated as two exploratory case studies, not as a controlled comparison of dataset types.

## 4. Methodology

### Predictive Performance

- Classification model: Logistic Regression
- Validation: Stratified 5-fold cross-validation
- Preprocessing: Standardization and PCA fitted within each training fold
- Baselines: Original features versus PCA retaining 95% explained variance
- Metrics: Accuracy, balanced accuracy, macro F1-score, ROC-AUC, and fold-to-fold variability

Additional experiments evaluate different explained-variance thresholds on the Breast Cancer dataset.

### PCA Subspace Stability

Subspace stability is evaluated using the mean squared cosine of principal angles between estimated principal subspaces.

The experiments examine:
- Different numbers of retained components
- Different sample fractions
- Clean versus Gaussian-noise-perturbed versions of the same sampled data

A stability similarity near 1 indicates that the compared subspaces are highly similar.

For visualization, stability loss is defined as:

Stability loss = 1 − mean subspace similarity

A larger stability loss indicates a greater change in the estimated subspace.

### Controlled Noise

Gaussian noise is introduced at levels 0.00, 0.05, 0.10, and 0.20 in standardized feature space.

These are controlled experimental perturbations, not measurements of naturally occurring noise in every real-world setting.

## 5. Preliminary Findings

### Breast Cancer Dataset

At the 95% explained-variance setting:

- Original features: 30 predictors, approximately 97.37% mean accuracy
- PCA: approximately 10 components, 97.72% mean accuracy

The observed difference is small and has not been established as statistically significant.

The variance-threshold experiment also shows that predictive performance does not necessarily improve monotonically as the explained-variance threshold increases.

Under controlled noise, the relative performance of PCA and the original features varies with noise intensity and metric.

### Wine Dataset

At the 95% explained-variance setting:

- Original features: 13 predictors, approximately 98.33% mean accuracy
- PCA: approximately 10 components, approximately 98.33% mean accuracy

PCA reduced dimensionality while preserving the observed mean accuracy in this experiment.

The high classification scores create a ceiling-effect concern, and the small dataset limits generalization.

### Stability Experiments

Across the tested configurations, subspace similarity generally declines as perturbation noise increases, with larger retained subspaces often showing greater sensitivity.

Larger sample fractions often improve stability, but the pattern is not perfectly monotonic for every component count and noise level.

These findings are exploratory. Similarity scores near 1 should be interpreted carefully, and small differences should not be overstated.

## 6. Project Structure

```text
pca_reliability_map/
├── data/                               # Local datasets; excluded from Git
├── outputs/                            # Generated results and plots; excluded from Git
├── baseline_experiment.py
├── baseline_real_dataset.py
├── breast_cancer_noise_performance_clean.py
├── breast_cancer_noise_stability_clean.py
├── build_reliability_map.py
├── inspect_dataset.py
├── inspect_wine_dataset.py
├── pca_noise_experiment.py
├── pca_stability_experiment.py
├── pca_threshold_experiment.py
├── wine_baseline_experiment.py
├── wine_noise_experiment.py
├── wine_noise_performance_clean.py
├── wine_noise_stability_experiment.py
├── wine_stability_experiment.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 7. Environment Setup

Python 3.10 or later is recommended.

Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## 8. Running the Experiments

Run the experiment scripts from the project root after placing the required datasets in `data/` or configuring the relevant dataset-fetching scripts.

Suggested order:

1. Inspect and validate the datasets.
2. Run the baseline experiments.
3. Run the PCA explained-variance threshold experiment.
4. Run the clean noise-performance experiments.
5. Run the subspace-stability experiments.
6. Run the comparative plotting script.

Example:

```powershell
python inspect_dataset.py
python inspect_wine_dataset.py
python breast_cancer_noise_performance_clean.py
python breast_cancer_noise_stability_clean.py
python wine_noise_performance_clean.py
python wine_noise_stability_experiment.py
python build_reliability_map.py
```

Some experiments may require other scripts to generate their input files first. Check each script's dataset paths and output filenames before running it independently.

Generated CSV files and figures are saved in `outputs/`.

## 9. Reproducibility Notes

- Dataset files and generated outputs are excluded from Git to keep the repository lightweight.
- Keep the dataset source and preprocessing steps documented.
- Results may vary slightly across software versions or random seeds.
- For a reproducible rerun, record the random seeds, dependency versions, and exact experiment configuration.
- Do not treat overlapping resampled subsets as statistically independent observations.

## 10. Limitations

- Only two real datasets are currently included.
- Gaussian noise is a controlled perturbation and may not represent every practical source of measurement error.
- The Wine dataset is small and has near-ceiling classification performance.
- Small differences in cross-validation metrics have not been established as statistically significant.
- Stability results depend on the chosen component count, sample fraction, perturbation method, and similarity metric.
- A validated, weighted composite reliability score has not yet been established.

## 11. Future Research

Potential extensions include:

- Testing more real datasets with varied dimensionality and sample sizes.
- Comparing PCA with robust PCA and other dimensionality-reduction methods.
- Testing alternative noise models and realistic missing-data mechanisms.
- Quantifying uncertainty in predictive performance and subspace stability.
- Investigating the effect of multicollinearity and near-equal eigenvalues.
- Developing and validating a composite PCA reliability framework on independent datasets.

## 12. References

1. Wolberg, W., Mangasarian, O., Street, N., & Street, N. (1993). *Breast Cancer Wisconsin (Diagnostic)*. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B
2. Aeberhard, S., Coomans, D., & de Vel (1992). *Wine recognition data*. UCI Machine Learning Repository. https://doi.org/10.24432/C5PC7J
3. Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: A review and recent developments. *Philosophical Transactions of the Royal Society A, 374*(2065), 20150202. https://doi.org/10.1098/rsta.2015.0202
