import json
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

def main():
    # Load tweets
    # data_filename = 'Ch6_data_tweets_Python_Combined110.json'
    data_filename = os.path.join(os.path.dirname(__file__), 'Ch6_data_tweets_Python_Combined110.json')
    print(f"Loading tweets from {data_filename}...")
    
    texts = []
    try:
        with open(data_filename, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    tweet = json.loads(line)
                    if 'text' in tweet:
                        texts.append(tweet['text'])
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print("Data file not found.")
        return

    print(f"Loaded {len(texts)} tweets.")
    
    if len(texts) == 0:
        print("No tweets found.")
        return

    # In a real scenario, these would be manually labeled.
    # For demonstration, we will label them using a simple keyword logic
    # "good", "great", "love", "awesome" -> Positive (1)
    # "bad", "hate", "terrible", "error" -> Negative (0)
    # else -> random or skipped? 
    # Let's create synthetic labels for all of them for the sake of the exercise
    
    labels = []
    labeled_texts = []
    
    positive_words = ["good", "great", "love", "awesome", "best", "happy", "nice"]
    negative_words = ["bad", "hate", "terrible", "worst", "sad", "error", "fail", "slow"]
    
    for text in texts:
        t_lower = text.lower()
        is_pos = any(w in t_lower for w in positive_words)
        is_neg = any(w in t_lower for w in negative_words)
        
        if is_pos and not is_neg:
            labels.append(1)
            labeled_texts.append(text)
        elif is_neg and not is_pos:
            labels.append(0)
            labeled_texts.append(text)
        else:
            # Ambiguous or neutral, let's just randomly assign for the sake of having a full dataset 
            # OR better, skip them to show "clean" data. 
            # But if we skip too many, we might have no data.
            # Let's default to 1 (Positive) for Python usually implies fans? No that's biased.
            # Let's just create a dummy label based on length mod 2 to ensure we can run the code
            # But the keyword match is better.
            pass
            
    # If we don't have enough labeled data from keywords, let's force some labels
    if len(labeled_texts) < 10:
        print("Not enough keyword-matches for robust testing. Generating random labels for demonstration.")
        labels = np.random.randint(0, 2, size=len(texts))
        labeled_texts = texts
    else:
        print(f"Labeled {len(labeled_texts)} tweets based on keywords.")
        
    X = labeled_texts
    y = np.array(labels)
    
    # Create transformer
    vectorizer = CountVectorizer(stop_words='english')
    X_counts = vectorizer.fit_transform(X)
    
    # Train Naive Bayes
    clf = MultinomialNB()
    
    # Cross validation
    if len(y) > 5:
        scores = cross_val_score(clf, X_counts, y, scoring='accuracy', cv=min(5, len(y)))
        print(f"Average Accuracy: {np.mean(scores)*100:.1f}%")
    else:
        clf.fit(X_counts, y)
        print("Not enough data for cross-validation. Trained on all data.")
        print("Model score:", clf.score(X_counts, y))
        
    # Show most informative features (if possible)
    clf.fit(X_counts, y)
    
    # Basic Feature Importance (Log probabilities)
    # Neg class = 0, Pos class = 1
    neg_class_prob_sorted = clf.feature_log_prob_[0, :].argsort()[::-1]
    pos_class_prob_sorted = clf.feature_log_prob_[1, :].argsort()[::-1]
    
    print("\nMost predictive words for Class 0 (Negative/Other):")
    print(np.take(vectorizer.get_feature_names_out(), neg_class_prob_sorted[:10]))
    
    print("\nMost predictive words for Class 1 (Positive):")
    print(np.take(vectorizer.get_feature_names_out(), pos_class_prob_sorted[:10]))

if __name__ == "__main__":
    main()
