import numpy as np
from collections import defaultdict

import os

def main():
    dataset_filename = os.path.join(os.path.dirname(__file__), "affinity_dataset.txt")
    X = np.loadtxt(dataset_filename)
    n_samples, n_features = X.shape
    
    print(f"This dataset has {n_samples} samples and {n_features} features")
    print(X[:5])
    
    # The features represent:
    # 0: Bread
    # 1: Milk
    # 2: Cheese
    # 3: Apples
    # 4: Bananas
    features = ["Bread", "Milk", "Cheese", "Apples", "Bananas"]
    
    # Find rules of the type: If X then Y
    # We count Occurrences of X (premise) and (X and Y) (conclusion)
    
    valid_rules = defaultdict(int)
    invalid_rules = defaultdict(int)
    num_occurences = defaultdict(int)
    
    # Iterate over each sample
    for sample in X:
        for premise in range(n_features):
            if sample[premise] == 0: continue
            
            num_occurences[premise] += 1
            
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
        confidence[rule] = valid_rules[rule] / num_occurences[premise]
        
    def print_rule(premise, conclusion, support, confidence, features):
        premise_name = features[premise]
        conclusion_name = features[conclusion]
        print(f"Rule: If a person buys {premise_name} they will also buy {conclusion_name}")
        print(f" - Support: {support}")
        print(f" - Confidence: {confidence:.3f}")

    print("\n--- All Rules ---")
    for premise, conclusion in confidence:
        print_rule(premise, conclusion, support[(premise, conclusion)], confidence[(premise, conclusion)], features)
        
    # Sort by support
    from operator import itemgetter
    sorted_support = sorted(support.items(), key=itemgetter(1), reverse=True)
    
    print("\n--- Top 5 Rules by Support ---")
    for index in range(5):
        (premise, conclusion) = sorted_support[index][0]
        print_rule(premise, conclusion, support[(premise, conclusion)], confidence[(premise, conclusion)], features)
        
    # Sort by confidence
    sorted_confidence = sorted(confidence.items(), key=itemgetter(1), reverse=True)
    
    print("\n--- Top 5 Rules by Confidence ---")
    for index in range(5):
        (premise, conclusion) = sorted_confidence[index][0]
        print_rule(premise, conclusion, support[(premise, conclusion)], confidence[(premise, conclusion)], features)

if __name__ == "__main__":
    main()
