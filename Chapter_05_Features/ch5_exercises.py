import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from scipy.stats import pearsonr
from sklearn.tree import DecisionTreeClassifier
try:
    from sklearn.model_selection import cross_val_score
except ImportError:
    from sklearn.cross_validation import cross_val_score # Fallback for older sklearn
from sklearn.base import TransformerMixin
from sklearn.utils import as_float_array
from sklearn.pipeline import Pipeline

def main():
    # Load data
    # Adjusted path to use the local file
    adult_filename = os.path.join(os.path.dirname(__file__), "Ch5_ExtractingFeatures_Chi2PearsonPCA_adult.data")
    
    if not os.path.exists(adult_filename):
        print(f"Error: {adult_filename} not found.")
        return

    adult = pd.read_csv(adult_filename, header=None, names=["Age", "Work-Class", "fnlwgt", "Education",
                                                            "Education-Num", "Marital-Status", "Occupation",
                                                            "Relationship", "Race", "Sex", "Capital-gain",
                                                            "Capital-loss", "Hours-per-week", "Native-Country",
                                                            "Earnings-Raw"])
    
    adult.dropna(how='all', inplace=True)
    
    print("--- Columns ---")
    print(adult.columns)
    
    print("\n--- Hours-per-week Description ---")
    print(adult["Hours-per-week"].describe())
    
    print("\n--- Education-Num Median ---")
    print(adult["Education-Num"].median())
    
    print("\n--- Work-Class Unique Values ---")
    print(adult["Work-Class"].unique())

    # Variance Threshold Example
    print("\n--- Variance Threshold Example ---")
    X = np.arange(30).reshape((10, 3))
    X[:,1] = 1
    print("X matrix:")
    print(X)
    
    vt = VarianceThreshold(threshold=2)
    Xt = vt.fit_transform(X)
    print("Transformed X (Xt):")
    print(Xt)
    print("Variances:")
    print(vt.variances_)

    # Feature Selection on Adult Data
    print("\n--- Feature Selection on Adult Data ---")
    X = adult[["Age", "Education-Num", "Capital-gain", "Capital-loss", "Hours-per-week"]].values
    # Note: The dataset might have leading spaces, so ' >50K' is likely correct if skipinitialspace=False (default)
    y = (adult["Earnings-Raw"] == ' >50K').values

    print("\n--- Chi2 Selection ---")
    transformer = SelectKBest(score_func=chi2, k=3)
    Xt_chi2 = transformer.fit_transform(X, y)
    print("Chi2 Scores:")
    print(transformer.scores_)
    print(np.sort(transformer.scores_))

    

    def multivariate_pearsonr(X, y):
        scores, pvalues = [], []
        for column in range(X.shape[1]):
            cur_score, cur_p = pearsonr(X[:,column], y)
            scores.append(abs(cur_score))
            pvalues.append(cur_p)
        return (np.array(scores), np.array(pvalues))

    print("\n--- Pearson Selection ---")
    transformer = SelectKBest(score_func=multivariate_pearsonr, k=3)
    Xt_pearson = transformer.fit_transform(X, y)
    print("Pearson Scores:")
    print(transformer.scores_)

    print("\n--- Classification Performance ---")
    clf = DecisionTreeClassifier(random_state=14)
    scores_chi2 = cross_val_score(clf, Xt_chi2, y, scoring='accuracy')
    scores_pearson = cross_val_score(clf, Xt_pearson, y, scoring='accuracy')
    print("Chi2 performance: {0:.3f}".format(scores_chi2.mean()))
    print("Pearson performance: {0:.3f}".format(scores_pearson.mean()))

    class MeanDiscrete(TransformerMixin):
        def fit(self, X, y=None):
            X = as_float_array(X)
            self.mean = np.mean(X, axis=0)
            return self

        def transform(self, X):
            X = as_float_array(X)
            assert X.shape[1] == self.mean.shape[0]
            return X > self.mean

    print("\n--- Mean Discrete Transformation ---")
    mean_discrete = MeanDiscrete()
    X_mean = mean_discrete.fit_transform(X)

    pipeline = Pipeline([('mean_discrete', MeanDiscrete()),
                         ('classifier', DecisionTreeClassifier(random_state=14))])
    scores_mean_discrete = cross_val_score(pipeline, X, y, scoring='accuracy')
    print("Mean Discrete performance: {0:.3f}".format(scores_mean_discrete.mean()))

if __name__ == "__main__":
    main()
