import re
import os
import csv
import random
import requests
import pandas as pd

from fake_useragent import UserAgent
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from pymongo import MongoClient

client = MongoClient('52.169.70.76', 27017)
db = client['emotion_twitter']
collection = db['tweets']

ua = UserAgent()
HEADERS_LIST = [ua.chrome, ua.google, ua['google chrome'], ua.firefox, ua.ff]


def clean_text(string):
    """
    Remove links, mentions, special characters and numbers
    """
    return ' '.join(re.sub("(@[A-Za-z]+)|([^A-Za-z \t])|(\w+:\/\/\S+)", "", string.lower()).split())


def get_tweet_from_id(id, sentiment):
    url = "https://twitter.com/dummy/status/" + id
    headers = {'User-Agent': random.choice(HEADERS_LIST)}
    try:
        response = requests.get(url, headers=headers)
        html = response.text
        soup = BeautifulSoup(html, 'lxml')
        tweet = soup.find(property='og:description').get('content')
        tweet = clean_text(tweet)
        collection.insert({'text': tweet, 'sentiment': sentiment})
    except:
        return


def parse_text_emotion(path):
    with open(path, mode='r') as f:
        for line in f:
            sentiment = line.split(',')[0]
            if sentiment in ['empty', 'sentiment', 'neutral']:
                continue
            tweet = clean_text(line.split(',')[-1])


def parse_wassa(path):
    """
    Parse WASSA-2017 text files
    :param path:
    :return: tuple of list of sentences, list of labels
    """
    tweets, labels = [], []
    lines = open(path).readlines()
    for line in lines:
        sentiment = line.split(',')[-1]
        text = line.split(',')[0]
        if len(text.split()) < 2:
            continue
        tweets.append(clean_text(text))
        labels.append(clean_text(sentiment))
    save_lists_as_csv((tweets, labels), 'data/text_emotion.csv')


def parse_20k_txt(path):
    tweets, labels = [], []
    lines = open(path).readlines()
    for line in lines:
        tweets.append(clean_text(line).rsplit(' ', 1)[0])
        labels.append(line.split()[-1])
    return tweets, labels


def build_tweet_id_mapping(path):
    files = [os.path.join(r, file) for r, d, f in os.walk(path) for file in f]
    for file in files:
        lines = open(file).readlines()
        with ThreadPoolExecutor(max_workers=20) as executor:
            for line in lines:
                id = line.split()[0]
                sentiment = clean_text(line.split()[-1])
                executor.submit(get_tweet_from_id, id, sentiment)


def build_sentiment_id_mapping(path):
    files = [os.path.join(r, file) for r, d, f in os.walk(path) for file in f]
    for file in files:
        lines = open(file).readlines()
        for line in lines:
            id = line.split()[0]
            sentiment = line.split()[-1]


def save_lists_as_csv(pairs, path):
    sentences, labels = pairs[0], pairs[1]
    with open(path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(sentences, labels))


def save_dict_as_csv(mapping):
    with open('twitter_emotion.csv', 'w') as f:
        writer = csv.writer(f)
        for key, value in mapping.items():
            writer.writerow([key, value])


def find_labels(path):
    file = open(path).readlines()
    labels = defaultdict(int)
    for line in file:
        labels[line.split(',')[1].strip()] += 1
    return labels


def remove_records(path):
    df = pd.read_csv(path)
    print(df.isnull().values.any())
    df['sentiment'] = df['sentiment'].replace({'happiness': 'joy'}, regex=True)
    # df = df[df.sentiment.isin(('happiness', 'anger', 'sadness', 'love', 'fear', 'thankfulness'))]
    df.to_csv('data/text_emotion2.csv', index=False)


if __name__ == "__main__":
    # parse_wassa('data/text_emotion.csv')
    remove_records('data/text_emotion.csv')
    # print(find_labels('data/text_emotion2.csv'))
