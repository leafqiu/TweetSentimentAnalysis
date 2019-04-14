# -*- coding: utf-8 -*-
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer


def clean_user(sentence):
    p_user = re.compile('@(\w+)')
    return re.sub(p_user, '', sentence)


def clean_tag(sentence):
    p_tag = re.compile('#(\w+)')
    return re.sub(p_tag, '', sentence)


def clean_link(sentence):
    p_link = re.compile('(http|https|ftp)://[a-zA-Z0-9\\./]+')
    return re.sub(p_link, '', sentence)


def clean_num(sentence):
    p_num = re.compile('[0-9]+')
    return re.sub(p_num, '', sentence)


def tokenize_clean(tweets):
    stops = stopwords.words('english')
    punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&',
                    '!', '*', '@', '#', '$', '%', '``', '\'\'', '...', '-', '/']
    stemmer = PorterStemmer()
    tidy_tweet = []
    for tweet in tweets:
        list_words = [word.lower() for word in word_tokenize(tweet)]
        filtered_words = [w for w in list_words if w not in stops and w not in punctuations]
        stem_words = [stemmer.stem(word) for word in filtered_words]
        tidy_tweet.append(' '.join(stem_words))
    return tidy_tweet


# clean the train and test data
# 1.remove @user, tag and links; 2.remove number and punctuation; 3.remove stopping words; 4.word stemming;
def clean(file, type):
    tidy_tweet = []
    polarity = []
    id = []
    for line in file.readlines():
        contents = line.split('\t')
        tidy_tweet.append(contents[2])
        polarity.append(contents[1])
        id.append(contents[0])
    tidy_tweet = map(clean_user, tidy_tweet)
    tidy_tweet = map(clean_tag, tidy_tweet)
    tidy_tweet = map(clean_link, tidy_tweet)
    tidy_tweet = map(clean_num, tidy_tweet)
    tidy_tweet = tokenize_clean(tidy_tweet)
    if type == 0:
        filename = 'tidy_train.csv'
    elif type == 2:
        filename = 'tidy_test.csv'
    else:
        filename = 'tidy_dev.csv'
    with open('data/'+filename, 'w', encoding='utf-8') as f:
        for index, tweet in enumerate(tidy_tweet):
            str = '\t'.join([id[index], polarity[index], tweet])
            f.write(str+'\n')


def main():
    train_file = open('data/twitter-2016train-A.txt', 'r', encoding='utf-8')
    dev_file = open('data/twitter-2016dev-A.txt', 'r', encoding='utf-8')
    test_file = open('data/twitter-2016devtest-A.txt', 'r', encoding='utf-8')
    clean(train_file, 0)
    clean(dev_file, 1)
    clean(test_file, 2)


if __name__ == '__main__':
    main()