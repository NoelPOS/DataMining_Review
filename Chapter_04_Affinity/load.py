import numpy as np
from collections import defaultdict

import os
dataset_filename = os.path.join(os.path.dirname(__file__), "affinity_dataset.txt")
X = np.loadtxt(dataset_filename)
n_features = X.shape[1]

valid_rules = defaultdict(int)
invalid_rules = defaultdict(int)
num_occurances = defaultdict(int)

for sample in X:
    for premise in range(n_features):
        if sample[premise] == 0: continue
        num_occurances[premise] += 1
        for conclusion in range(n_features):
            if premise == conclusion: continue
            if sample[conclusion] == 1:
                valid_rules[(premise, conclusion)] += 1
            else:
                invalid_rules[(premise, conclusion)] += 1

support = valid_rules

confidence = defaultdict(float)
for premise, conclusion in valid_rules.keys():
    rule = (premise, conclusion)
    confidence[rule] = valid_rules[rule] / num_occurances[premise]

def print_rule(premise, conclusion, support, confidence, features):
    premise_name = features[premise]
    conclusion_name = features[conclusion]
    print("Rule: If a person buys {0} they will also buy {1}".format(premise_name, conclusion_name))
    print(" - Support: {0}".format(support[(premise, conclusion)]))
    print(" - Confidence: {0:.3f}".format(confidence[(premise, conclusion)]))

features = ["bread", "milk", "cheese", "apples", "bananas"]

for premise, conclusion in confidence.keys():
    print_rule(premise, conclusion, support, confidence, features)