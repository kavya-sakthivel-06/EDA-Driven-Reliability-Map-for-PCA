
## Experiments Completed

### 1. Real-Dataset Baseline
- Dataset: Breast Cancer Wisconsin (Diagnostic), UCI ML Repository
- Observations: 569
- Features: 30
- Classification model: Logistic Regression
- Validation: Stratified 5-fold cross-validation
- Comparison: Original features versus PCA retaining 95% variance

### 2. PCA Variance Threshold Experiment
Compared the original features against PCA retaining 80%, 85%, 90%, 95%, and 99% of variance.

Metrics:
- Accuracy
- Balanced accuracy
- F1-score
- ROC-AUC
- Fold-to-fold variability
- Number of principal components

### Preliminary Findings
On the initial dataset, PCA retaining 95% variance
achieved 97.72% mean accuracy using 10 components,
compared with 97.37% using the original 30 features.

The findings are preliminary and require further stability
analysis and validation on additional real datasets.

## Next Steps
- [ ] Evaluate PCA subspace stability under resampling
- [ ] Validate on another real dataset
- [ ] Integrate performance and stability into a reliability map


### 3. PCA Subspace Stability Experiment

- Dataset: UCI Breast Cancer Wisconsin (Diagnostic)
- Sample fractions: 70%, 80%, and 90%
- PCA component counts: 5, 10, and 17
- Resampling repetitions: 20 per sample fraction
- Pairwise comparisons: 190 per component count and sample fraction
- Stability metric: Mean squared cosine of principal angles

Preliminary observation:
Subspace similarity increased with sample size for all
three component counts in this experiment. The 5-component
subspace had the highest similarity at every sample size.

These results are exploratory and require validation on
additional datasets and under other perturbations.


### 4. PCA Reliability Under Controlled Noise

Noise levels tested: 0.00, 0.05, 0.10, and 0.20
Classification: Logistic Regression with stratified 5-fold CV
PCA threshold: 95% explained variance
Stability: 20 random subsets at 80% sample size

Preliminary observations:
- PCA had slightly higher mean accuracy than the original
  features at all four tested noise levels.
- PCA subspace similarity remained high, but did not
  change monotonically with noise.
- The number of components required to retain 95% variance
  increased as noise intensity increased.

Limitations:
- Noise levels are controlled experimental settings.
- Only one real dataset has been studied so far.
- Small metric differences are not established as significant.
- The stability comparisons use overlapping subsets.

Generated results:
- outputs/pca_noise_performance.csv
- outputs/pca_noise_stability.csv
