# Chapter 3: Sports Analytics with Decision Trees

## 1. The Concepts
### Decision Trees
*   Imagine a flowchart: "Did they win the last game?" -> Yes/No. "Is the opponent strong?" -> Yes/No.
*   The algorithm learns these questions automatically to split the data into pure groups (e.g., all Wins in one group).

### Random Forests
*   A "Forest" is made of many Trees.
*   Each tree gets a random subset of data and features.
*   They vote on the answer. This reduces the risk of a single tree being "wrong" or obsessed with a weird pattern (overfitting).

## 2. The Dataset: NBA 2013-2014
*   **Rows**: Basketball games.
*   **Features**: Date, Start Time, Visitor Team, Home Team, Scores.
*   **Goal**: Predict `HomeWin` (True/False).

## 3. Code Walkthrough (`ch3.py`)

### Step 1: Feature Engineering
Raw data isn't enough. We need *context*.
```python
# Create a feature 'HomeLastWin': Did the home team win their previous match?
won_last = defaultdict(int)
for index, row in dataset.sort_values(by="Date").iterrows():
    home_team = row["Home_Team"]
    dataset.loc[index, "HomeLastWin"] = int(won_last[home_team])
    won_last[home_team] = row["HomeWin"] # Update for next time
```

### Step 2: Decision Tree Model
```python
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=14)
# Train on our engineered features
X_previouswins = dataset[["HomeLastWin", "VisitorLastWin"]].values
scores = cross_val_score(clf, X_previouswins, y_true, scoring='accuracy')
```

### Step 3: Random Forest & Grid Search
We use `GridSearchCV` to test many combinations of settings (parameters) automatically.
```python
parameter_space = {
    "max_features": [2, 10, 'auto'],
    "n_estimators": [100,], # Number of trees
    "min_samples_leaf": [2, 4, 6],
}
grid = GridSearchCV(clf_rf, parameter_space, scoring='accuracy')
grid.fit(X_all, y_true)
print(grid.best_estimator_)
```

## 4. Key Takeaway
Feature Engineering (creating meaningful variables like "Last Win") is often more important than the choice of algorithm. A Random Forest usually beats a single Decision Tree.
