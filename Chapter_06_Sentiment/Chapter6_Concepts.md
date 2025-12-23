# Chapter 6: Sentiment Analysis (Text Mining)

## 1. The Concept: Naive Bayes
Text is just words. How do we do math on words?
*   **Bag of Words**: We count word frequencies. "I happy" -> {I:1, happy:1}.
*   **Naive Bayes**: A probabilistic classifier based on Bayes' Theorem.
    *   It calculates: `P(Positive | "Happy")`.
    *   It is "Naive" because it assumes words are independent (it ignores grammar/order), yet it works surprisingly well for text.

## 2. The Dataset: Twitter Data
*   Collected via Twitter API.
*   JSON format containing text of tweets.
*   **Task**: Label them as Positive (1) or Negative (0).

## 3. Code Walkthrough (`ch6_sentiment.py`)

### Step 1: Preprocessing & Labeling
Since we don't have human labels, we simulate them with keywords (in a real project, you'd use a labelled dataset).
```python
# Simple keyword logic
if "good" in text: label = 1
elif "bad" in text: label = 0
```

### Step 2: Vectorization
Turning text strings into a number matrix.
```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words='english') 
# stop_words='english' removes "the", "and", "is"...
X_counts = vectorizer.fit_transform(X)
```

### Step 3: Model Training
```python
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_counts, y)
```

### Step 4: Inspecting The Brain
We can ask the model which words are most "Positive".
```python
# Sort features by their probability for Class 1 (Positive)
pos_class_prob_sorted = clf.feature_log_prob_[1, :].argsort()[::-1]
top_words = np.take(vectorizer.get_feature_names_out(), pos_class_prob_sorted[:10])
print(top_words)
```

## 4. Key Takeaway
For text analysis, **Preprocessing** (cleaning URLs, removing stop words) and **Vectorization** (Bag of Words or TF-IDF) are the critical steps before feeding data into a model like Naive Bayes.
