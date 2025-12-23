# Chapter 5: Feature Selection & PCA

## 1. The Concept: Dimensionality Reduction
Real-world data is messy and has too many columns (features).
*   **Feature Selection**: Filtering out useless columns (e.g., purely random noise).
*   **PCA (Principal Component Analysis)**: A mathematical transformation that merges correlated columns into new "Super Columns" (Principal Components) that explain the variance (spread) of the data.

## 2. The Dataset: Adult Census
*   **Source**: Census data.
*   **Task**: Predict if income is `>50K` or `<=50K`.
*   **Features**: Age, Education, Marital Status, Occupation, etc.

## 3. Code Walkthrough (`ch5.py`)

### Step 1: SelectKBest (Chi-Squared)
This statistical test measures the dependency between a feature and the target.
```python
from sklearn.feature_selection import SelectKBest, chi2
# Pick the top k=5 features that range most with Income
chi2_selector = SelectKBest(score_func=chi2, k=5)
X_kbest = chi2_selector.fit_transform(X, y)
```

### Step 2: PCA (Principal Component Analysis)
First, we **Standardize** the data (Scale it) so large numbers (Salary) don't dominate small numbers (Age).
```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)
```

### Step 3: Explained Variance
We check how much "information" (variance) each new component holds.
```python
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
# How many components to explain 95% of data?
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
```

## 4. Key Takeaway
More data is not always better. "Curse of Dimensionality" means too many features can confuse a model. PCA helps compress data while keeping the important signals.
