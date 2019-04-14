# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def word2label(word):
    if word == 'negative':
        return 0
    elif word == 'positive':
        return 1
    else:
        return 2


def get_feature(data, vectorizer):
    # print(vectorizer.get_feature_names()[:10])
    text = list(data.iloc[:, 2])
    label = list(data.iloc[:, 1])
    # print(label[:10])
    label = list(map(word2label, label))
    # print(label[:10])
    feature = vectorizer.transform(text).toarray()
    df1 = pd.DataFrame(feature)
    df2 = pd.DataFrame(label)
    return pd.concat([df1, df2], axis=1)


# extract unigram as feature
def unigram_feature(train, test):
    unigram_vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=1)
    corpus = list(train.iloc[:, 2])
    unigram_vectorizer.fit(corpus)
    feature_train = get_feature(train, unigram_vectorizer)
    print('unigram shape:', feature_train.shape)
    feature_test = get_feature(test, unigram_vectorizer)
    return feature_train, feature_test
    # feature_train.to_csv('data/train_unigram.csv', index=False, header=None)
    # feature_test.to_csv('data/test_unigram.csv', index=False, header=None)


def bigram_feature(train, test, max_feature=None):
    bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=max_feature, min_df=1)
    corpus = list(train.iloc[:, 2])
    bigram_vectorizer.fit(corpus)
    feature_train = get_feature(train, bigram_vectorizer)
    print('bigram shape:', feature_train.shape)
    feature_test = get_feature(test, bigram_vectorizer)
    return feature_train, feature_test
    # feature_train.to_csv('data/train_bigram.csv', index=False, header=None)
    # feature_test.to_csv('data/test_bigram.csv', index=False, header=None)


def unigram_bigram_feature(train, test, max_feature=None):
    uni_bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=max_feature, min_df=1)
    corpus = list(train.iloc[:, 2])
    uni_bigram_vectorizer.fit(corpus)
    feature_train = get_feature(train, uni_bigram_vectorizer)
    print('unigram+bigram shape:', feature_train.shape)
    feature_test = get_feature(test, uni_bigram_vectorizer)
    return feature_train, feature_test


def main():
    train = pd.read_csv('data/tidy_train.csv', sep='\t', header=None)
    dev = pd.read_csv('data/tidy_dev.csv', sep='\t', header=None)
    test = pd.read_csv('data/tidy_test.csv', sep='\t', header=None)
    train = train.append(dev, ignore_index=True)
    # print(train.shape)
    train_unigram, test_unigram = unigram_feature(train, test)
    train_unigram.to_csv('data/train_unigram.csv', index=False, header=None)
    test_unigram.to_csv('data/test_unigram.csv', index=False, header=None)
    train_bigram, test_bigram = bigram_feature(train, test)
    train_bigram.to_csv('data/train_bigram.csv', index=False, header=None)
    test_bigram.to_csv('data/test_bigram.csv', index=False, header=None)


if __name__ == '__main__':
    main()