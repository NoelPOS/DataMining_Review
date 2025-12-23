# Chapter 1: Introduction to Data Mining

## 1. The Concepts
### What is Data Mining?
It is the process of discovering patterns in large data sets involving methods at the intersection of machine learning, statistics, and database systems.

### Scikit-Learn (sklearn)
The most popular Python library for Machine Learning. It contains:
*   **Datasets**: Toy data to practice with (like Iris, Boston Housing).
*   **Models**: Algorithms like Decision Trees, SVM, etc.
*   **Metrics**: Tools to measure accuracy.

## 2. The Code Walkthrough

### `hello.py`
A simple sanity check to ensure Python is running.
```python
print("Hello World")
```

### `sk_practice.py`
This script demonstrates loading a built-in dataset from Scikit-Learn.
```python
from sklearn.datasets import load_iris

# Load the famous Iris flower dataset
dataset = load_iris()
X = dataset.data   # The features (petal length, width, etc.)
y = dataset.target # The class (Setosa, Versicolor, Virginica)

print(X)
print(y)
```

## 3. Key Takeaway
Before building complex models, always verify your environment works and you can load data into `X` (Features) and `y` (Labels).
