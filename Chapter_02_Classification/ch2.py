import os
import csv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

data_folder = os.path.dirname(__file__)
data_filename = os.path.join(data_folder, "IonosphereData_NNAlgoClassify.data")

X = np.zeros((351, 34), dtype='float')
y = np.zeros((351,), dtype='bool')

try:
    with open(data_filename, 'r') as input_file:
        reader = csv.reader(input_file)
        for i, row in enumerate(reader):
            data = [float(datum) for datum in row[:-1]]
            X[i] = data
            y[i] = row[-1] == 'g'
except FileNotFoundError:
    print(f"Error: Data file not found at {data_filename}")
    print("Please make sure 'IonosphereData_NNAlgoClassify.data' is in the same directory.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=14)

print("--- Dataset Info ---")
print(f"There are {X_train.shape[0]} samples in the training dataset")
print(f"There are {X_test.shape[0]} samples in the testing dataset")
print(f"Each sample has {X_train.shape[1]} features")
print("-" * 20)

estimator = KNeighborsClassifier()
estimator.fit(X_train, y_train)

y_predicted = estimator.predict(X_test)
accuracy = np.mean(y_test == y_predicted) * 100
print(f"Initial accuracy (n_neighbors=5) on test set: {accuracy:.1f}%")
print("-" * 20)

scores = cross_val_score(estimator, X, y, scoring='accuracy')
average_accuracy = np.mean(scores) * 100
print(f"Average accuracy (n_neighbors=5) with cross-validation: {average_accuracy:.1f}%")
print("-" * 20)

avg_scores = []
all_scores = []
parameter_values = list(range(1, 21))

print(f"Running cross-validation for n_neighbors 1 to {len(parameter_values)}...")
for n_neighbors in parameter_values:
    estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_val_score(estimator, X, y, scoring='accuracy')
    avg_scores.append(np.mean(scores))
    all_scores.append(scores)
print("...Done.")

print("--- Preprocessing Example ---")

X_broken = np.array(X)
X_broken[:, ::2] /= 10


estimator = KNeighborsClassifier()
original_scores = cross_val_score(estimator, X, y, scoring='accuracy')
print(f"The original average accuracy is {np.mean(original_scores) * 100:.1f}%")

broken_scores = cross_val_score(estimator, X_broken, y, scoring='accuracy')
print(f"The 'broken' average accuracy is {np.mean(broken_scores) * 100:.1f}%")

# Fix with preprocessing
X_transformed = MinMaxScaler().fit_transform(X_broken)
transformed_scores = cross_val_score(estimator, X_transformed, y, scoring='accuracy')
print(f"The average accuracy after MinMaxScaler is {np.mean(transformed_scores) * 100:.1f}%")


scaling_pipeline = Pipeline([
    ('scale', MinMaxScaler()),
    ('predict', KNeighborsClassifier())
])
pipeline_scores = cross_val_score(scaling_pipeline, X_broken, y, scoring='accuracy')
print(f"The pipeline average accuracy is {np.mean(pipeline_scores) * 100:.1f}%")
print("-" * 20)

print("Generating plots...")

# First plot: Average accuracy vs n_neighbors
plt.figure(figsize=(10, 6))
plt.plot(parameter_values, avg_scores, '-o')
plt.title('Average Accuracy vs. n_neighbors')
plt.xlabel('n_neighbors (K)')
plt.ylabel('Average Accuracy')
plt.grid(True)

# Second plot: All cross-validation scores vs n_neighbors
plt.figure(figsize=(10, 6))
plt.plot(parameter_values, all_scores, 'bx')
plt.title('All Cross-Validation Scores vs. n_neighbors')
plt.xlabel('n_neighbors (K)')
plt.ylabel('Accuracy')
plt.grid(True)

# Additional plots from the book (using defaultdict)
from collections import defaultdict

all_scores = defaultdict(list)
parameter_values = list(range(1, 21))  # Including 20

for n_neighbors in parameter_values:
    estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_val_score(estimator, X, y, scoring='accuracy', cv=10)
    all_scores[n_neighbors].append(scores)

for parameter in parameter_values:
    scores = all_scores[parameter]
    n_scores = len(scores)
    plt.plot([parameter] * n_scores, scores, '-o')

plt.show()    