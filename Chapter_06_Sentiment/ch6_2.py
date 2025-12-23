import json
from collections import Counter

import os

# Read tweets from JSON file
tweets_text = []
data_filename = os.path.join(os.path.dirname(__file__), 'Ch6_data_tweets_Python_Combined110.json')
with open(data_filename, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:  # Skip empty lines
            try:
                tweet = json.loads(line)
                if 'text' in tweet:
                    tweets_text.append(tweet['text'].lower())
            except json.JSONDecodeError:
                continue  # Skip malformed lines

# Combine all tweet texts
all_text = ' '.join(tweets_text)

# Define stop words
stop_words = {
    "a", "an", "the", "for", "in", "on", "under", "to", "of",
    "and", "where", "all", "his", "them", "their", "one", "lie",
    "with", "by", "at", "into", "from", "as", "is", "are", "be",
    "this", "that", "it", "or", "you", "we", "i", "my", "your",
    "has", "have", "can", "will", "do", "does", "did", "been",
    "am", "if", "so", "but", "not", "no", "what", "who", "when",
    "how", "why", "now", "our", "us", "up", "out", "about", "more",
    "like", "get", "new", "use", "just", "make", "also", "here",
    "there", "than", "then", "these", "those", "such", "some",
    "amp", "rt", "via", "today", "day", "https", "http", "com",
    "httpstco", "httpstcoht7i62beyv"
}

# Split into words and filter
words = []
for word in all_text.split():
    # Remove punctuation and URLs
    clean_word = ''.join(c for c in word if c.isalnum())
    # Filter out URLs, stop words, and words less than 3 characters
    if (clean_word and 
        clean_word not in stop_words and 
        len(clean_word) > 2 and
        not clean_word.startswith('http')):
        words.append(clean_word)

# Count word frequencies
c = Counter(words)

print("="*60)
print("Top 20 Most Common Words in Python Tweets:")
print("="*60)
for word, count in c.most_common(20):
    print(f"{word:20s} : {count:3d}")
print("="*60)