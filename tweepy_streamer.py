from tweepy import API
from tweepy import Cursor
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

from textblob import TextBlob

import re
import twitter_credentials as tc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TwitterClient():
    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth)
        self.twitter_user = twitter_user

    def get_twitter_client_api(self):
        return self.twitter_client

    def get_user_timeline_tweets(self, num_tweets):
        tweets = []
        for tweet in Cursor(self.twitter_client.user_timeline, id=self.twitter_user).items(num_tweets):
            tweets.append(tweet)
        return tweets

    def get_friend_list(self, num_friends):
        friend_list = []
        for friend in Cursor(self.twitter_client.friends, id=self.twitter_user).items(num_friends):
            friend_list.append(friend)
        return friend_list

    def get_home_timeline_tweets(self, num_tweets):
        home_timeline_tweets = []
        for tweet in Cursor(self.twitter_client.home_timeline, id=self.twitter_user).items(num_tweets):
            home_timeline_tweets.append(tweet)
        return home_timeline_tweets
    
    def search_tweets(self, num_tweets, search_terms):
        query = " OR ".join(search_terms)
        query = query + "-filter:retweets"
        tweets = api.search(q=query, count=num_tweets, lang="en")

        return tweets

# Auth class


class TwitterAuthenticator():
    def authenticate_twitter_app(self):
        auth = OAuthHandler(tc.API_KEY, tc.API_SECRET_KEY)
        auth.set_access_token(tc.ACCESS_TOKEN, tc.ACCESS_SECRET_TOKEN)
        return auth


class TwitterStreamer():
    def __init__(self):
        self.twitter_auth = TwitterAuthenticator()

    def stream_tweets(self, fetched_tweets_filename, hash_tag_list):
        listener = TwitterListener(
            fetched_tweets_filename=fetched_tweets_filename)
        auth = self.twitter_auth.authenticate_twitter_app()

        stream = Stream(auth, listener)
        stream.filter(track=hash_tag_list)


class TwitterListener(StreamListener):
    def __init__(self, fetched_tweets_filename):
        self.fetched_tweets_filename = fetched_tweets_filename

    def on_data(self, data):
        try:
            print(data)
            with open(self.fetched_tweets_filename, 'a') as tf:
                tf.write(data)
            return True
        except BaseException as e:
            print(f"Error: {e}")
            return False

    def on_error(self, status):
        print(status)
        if status == 420:
            return False


class TweetAnalyzer():
    def tweets_to_data_frame(self, tweets):
        df = pd.DataFrame(
            data=[tweet.text for tweet in tweets], columns=["Tweets"])
        with open('tweets.json', 'w') as tf:
            tf.write(df.to_json())
        return df

    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def analyze_sentiment(self, tweet):
        analysis = TextBlob(self.clean_tweet(tweet))
        polarity = analysis.sentiment.polarity
        if polarity > 0:
            return 1
        elif polarity == 0:
            return 0
        return -1


if __name__ == "__main__":
    # hash_tag_list = ['homestuck', 'undertale', 'fnf']
    # fetched_tweets_filename = "tweets.json"

    twitter_client = TwitterClient()
    tweet_analyzer = TweetAnalyzer()

    api = twitter_client.get_twitter_client_api()

    search_terms = ["vaccine", "sinovac", "pfizer",
                    "astrazeneca", "moderna", "curevac", "sinopharm", "novavax"]
    tweets = twitter_client.search_tweets(num_tweets=100, search_terms=search_terms)

    df = tweet_analyzer.tweets_to_data_frame(tweets)
    df['sentiment'] = np.array(
        [tweet_analyzer.analyze_sentiment(tweet) for tweet in df['Tweets']]
    )

    tally = {"positive": 0, "negative": 0, "neutral": 0}
    for sentiment in df['sentiment']:
        if sentiment == 1:
            tally["positive"] += 1
        elif sentiment == 0:
            tally["neutral"] += 1
        else:
            tally["negative"] += 1

    print(tally)
    print(df.head(10))

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.axis('equal')
    polarities = ["Positive", "Negative", "Neutral"]
    tallies = [tally["positive"], tally["negative"], tally["neutral"]]
    ax.pie(tallies, labels=polarities, autopct='%1.2f%%')
    plt.show()
