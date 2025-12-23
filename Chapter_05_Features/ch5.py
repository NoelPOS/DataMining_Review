import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

def main():
    # Define columns based on adult.names
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income"
    ]

    # Load data
    data_path = os.path.join(os.path.dirname(__file__), "Ch5_ExtractingFeatures_Chi2PearsonPCA_adult.data")
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path, header=None, names=columns, skipinitialspace=True)
    except FileNotFoundError:
        print(f"Error: File {data_path} not found.")
        return

    print(f"Original shape: {df.shape}")

    # Handle missing values
    # Replace '?' with NaN and drop rows with missing values
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    print(f"Shape after dropping missing values: {df.shape}")

    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    # Split features and target
    X = df.drop('income', axis=1)
    y = df['income']

    # 1. Chi-Square Feature Selection
    print("\n--- Chi-Square Feature Selection ---")
    k = 5
    chi2_selector = SelectKBest(score_func=chi2, k=k)
    X_kbest = chi2_selector.fit_transform(X, y)

    print(f"Top {k} features selected by Chi-Square:")
    selected_indices = chi2_selector.get_support(indices=True)
    # Get the names of the selected columns
    selected_features = [columns[i] for i in selected_indices]
    for feature in selected_features:
        print(f"- {feature}")

    # 2. Pearson Correlation
    print("\n--- Pearson Correlation ---")
    # Calculate correlation matrix
    correlation_matrix = df.corr()
    # Get correlation with target 'income'
    target_correlation = correlation_matrix['income'].drop('income')
    print("Pearson Correlation with 'income' (sorted):")
    print(target_correlation.sort_values(ascending=False))

    # 3. PCA (Principal Component Analysis)
    print("\n--- PCA (Principal Component Analysis) ---")
    # Standardize the data first (important for PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    print("Explained Variance Ratio by Principal Component:")
    print(pca.explained_variance_ratio_)
    
    print("\nCumulative Explained Variance:")
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    print(cumulative_variance)
    
    # Find number of components for 95% variance
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"\nNumber of components needed to explain 95% variance: {n_components_95}")

if __name__ == "__main__":
    main()
