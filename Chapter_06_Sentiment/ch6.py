import tweepy
import os
import json

# Your keys
consumer_key = "fruMgv1sAI2IHhOYbGDu28t8u"
consumer_secret = "MRECVaUCIEWyBtcLH13VPFbzT0DLHg43INwKVX5zgUIT3Bk7U1"
access_token = "199886778420780258D-uQqj3YVJwk3XTS8Gc3H4M5PL5XWi3c"
access_token_secret = "IZ6ejYCJ1FaPeYFDnkqcLeccrs67rBD3Szs1lDCqorVGD"
bearer_token = "AAAAAAAAAAAAAAAAAAAAAAPSc5gEAAAAA7cukvSDnQKTuhP89AsjQcQD8gRR3DAOD3N8QXFehlFVgC4eXxJP21VVAg1lh6XFiaYlh7LQ61hQtEEDG"

client = tweepy.Client(
    bearer_token=bearer_token,
    consumer_key=consumer_key,
    consumer_secret=consumer_secret,
    access_token=access_token,
    access_token_secret=access_token_secret
)

# Output file, wait for 15 mins before downloading another 10 or 20 sets of data, max 180.
#output_filename = r'D:\ABAC\CSX.ITX.4202_DataMining\PythonCodes\Ch6_SocialMediaTwitter_NaiveBayes\data_tweets_Python.json'
#output_filename = r'D:\ABAC\CSX.ITX.4202_DataMining\PythonCodes\Ch6_SocialMediaTwitter_NaiveBayes\data_tweets_DonaldTrump.json'
# output_filename = r'D:\ABAC\CSX.ITX.4202_DataMining\PythonCodes\Ch6_SocialMediaTwitter_NaiveBayes\data_tweets_ElonMusk.json'
output_filename = os.path.join(os.path.dirname(__file__), 'data_tweets_ElonMusk.json')
print(output_filename)

#query = "python -is:retweet" #*2
#query = "Donald Trump -is:retweet" #*2
query = "Elon Musk -is:retweet" #*3

tweets = client.search_recent_tweets(query=query, max_results=20)
with open(output_filename, "a") as f:
    for tweet in tweets.data:
        f.write(json.dumps(tweet.data))
        f.write('\n\n')
print("Saved tweets!")