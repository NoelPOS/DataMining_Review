# Chapter 4: Affinity Analysis (Association Rules)

## 1. The Concept: Association Rules
This is often called **Market Basket Analysis**. We want to find rules like:
> "If a customer buys X, they are likely to buy Y."

Two key metrics measure this:
1.  **Support**: The frequency of the rule. (How many people bought *both* X and Y?).
2.  **Confidence**: The reliability of the rule. (Out of everyone who bought X, what % also bought Y?).

## 2. The Dataset: Matrix
*   A text file (`affinity_dataset.txt`) where rows are transactions and columns are items (Bread, Milk, Cheese, etc.).
*   `1` means bought, `0` means not bought.

## 3. Code Walkthrough (`ch4.py`)

### Step 1: Counting Occurrences
We loop through every transaction to count how often items appear together.
```python
for sample in X:
    for premise in range(n_features):
        if sample[premise] == 0: continue # Skip if they didn't buy the premise item
        
        num_occurences[premise] += 1
        
        for conclusion in range(n_features):
            if premise == conclusion: continue
            
            if sample[conclusion] == 1:
                valid_rules[(premise, conclusion)] += 1 # Found (X and Y)
            else:
                invalid_rules[(premise, conclusion)] += 1
```

### Step 2: Calculating Confidence
```python
confidence = defaultdict(float)
for premise, conclusion in valid_rules.keys():
    rule = (premise, conclusion)
    # Confidence = (X and Y) / (X)
    confidence[rule] = valid_rules[rule] / num_occurences[premise]
```

### Step 3: Ranking
We sort the rules to find the most useful ones. High confidence means strong predictive power.
```python
sorted_confidence = sorted(confidence.items(), key=itemgetter(1), reverse=True)
for index in range(5):
    print_rule(...)
```

## 4. Key Takeaway
This technique is used by Amazon/Netflix for "Customers who bought this also bought...". It doesn't predict a class/number, it discovers **relationships**.
