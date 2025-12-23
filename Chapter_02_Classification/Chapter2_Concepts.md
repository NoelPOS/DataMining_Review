# Chapter 2: Classification with K-Nearest Neighbors

## 1. The Concept: K-Nearest Neighbors (KNN)
**Classification** is the task of predicting a category (class) for a data point.
**KNN** is a "lazy learning" algorithm. It doesn't build a complex model. Instead, it memorizes the training data.
*   **How it works**: When you want to classify a new point, KNN looks at the 'K' closest points in the training set.
*   **Distance**: It usually uses Euclidean distance (straight line) to measure "closeness".
*   **Voting**: If K=3, and the 3 closest neighbors are [Good, Good, Bad], the algorithm predicts **Good** (2 vs 1).

## 2. The Dataset: Ionosphere
*   **Source**: Radar data collecting reflections from the ionosphere.
*   **Features**: 34 continuous numerical values (float).
*   **Target**: 'g' (Good) or 'b' (Bad).

## 3. Code Walkthrough (`ch2.py`)

### Step 1: Loading Data
```python
# We use standard CSV library to read the file
with open(data_filename, 'r') as input_file:
    reader = csv.reader(input_file)
    for i, row in enumerate(reader):
        data = [float(datum) for datum in row[:-1]] # Features
        X[i] = data
        y[i] = row[-1] == 'g' # Target (True if 'g')
```

### Step 2: Training and Testing
We cannot test on the same data we trained on (cheating). We split the data.
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=14)
```

### Step 3: The Model
```python
from sklearn.neighbors import KNeighborsClassifier
estimator = KNeighborsClassifier() # Default K=5
estimator.fit(X_train, y_train)    # Learn the patterns
```

### Step 4: Evaluation
We verify accuracy by seeing if `predicted == actual`.
```python
y_predicted = estimator.predict(X_test)
accuracy = np.mean(y_test == y_predicted) * 100
```

### Step 5: Optimization (Finding 'K')
The script loops through K=1 to K=20 to see which works best.
```python
for n_neighbors in parameter_values:
    estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_val_score(estimator, X, y, scoring='accuracy')
    # ... stores scores to plot them later
```

## 4. Key Takeaway
KNN is simple and effective for small datasets but gets slow with large data (because it has to measure distance to *every* point).
