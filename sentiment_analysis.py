# -*- coding: utf-8 -*-
import features as ft
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *


def classify_score(test_target, labels):
    average = 'macro'
    precision = precision_score(test_target, labels, average=average)
    recall = recall_score(test_target, labels, average=average)
    f1 = f1_score(test_target, labels, average=average)
    return precision, recall, f1


def naive_bayes(train_data, train_target, test_data, test_target):
    model = MultinomialNB()
    model.fit(train_data, train_target)
    labels = model.predict(test_data)
    # precision, recall, f1 = classify_score(test_target, labels)
    return classify_score(test_target, labels)


def logistic_regression(train_data, train_target, test_data, test_target):
    model = LogisticRegression()
    model.fit(train_data, train_target)
    labels = model.predict(test_data)
    return classify_score(test_target, labels)


# learning models to classify message's polarity
def main():
    train_text = pd.read_csv('data/tidy_train.csv', sep='\t', header=None)
    dev_text = pd.read_csv('data/tidy_dev.csv', sep='\t', header=None)
    test_text = pd.read_csv('data/tidy_test.csv', sep='\t', header=None)
    train_text = train_text.append(dev_text, ignore_index=True)

    # use unigrams as feature
    (train, test) = ft.unigram_feature(train_text, test_text)
    train_data = train.iloc[:, :-1]
    train_target = train.iloc[:, -1]
    test_data = test.iloc[:, :-1]
    test_target = test.iloc[:, -1]
    nb_result = naive_bayes(train_data, train_target, test_data, test_target)
    print('NaiveBayes+unigrams (precision, recall, f1_score):', nb_result)
    lr_result = logistic_regression(train_data, train_target, test_data, test_target)
    print('LogisticRegression+unigrams (precision, recall, f1_score):', lr_result)

    # use bigrams as feature
    (train, test) = ft.bigram_feature(train_text, test_text)
    train_data = train.iloc[:, :-1]
    train_target = train.iloc[:, -1]
    test_data = test.iloc[:, :-1]
    test_target = test.iloc[:, -1]
    nb_result = naive_bayes(train_data, train_target, test_data, test_target)
    print('NaiveBayes+bigrams (precision, recall, f1_score):', nb_result)
    lr_result = logistic_regression(train_data, train_target, test_data, test_target)
    print('LogisticRegression+bigrams (precision, recall, f1_score):', lr_result)

    # use unigrams+bigrams as feature
    (train, test) = ft.unigram_bigram_feature(train_text, test_text)
    train_data = train.iloc[:, :-1]
    train_target = train.iloc[:, -1]
    test_data = test.iloc[:, :-1]
    test_target = test.iloc[:, -1]
    nb_result = naive_bayes(train_data, train_target, test_data, test_target)
    print('NaiveBayes+unigrams+bigrams (precision, recall, f1_score):', nb_result)
    lr_result = logistic_regression(train_data, train_target, test_data, test_target)
    print('LogisticRegression+unigrams+bigrams (precision, recall, f1_score):', lr_result)


if __name__ == '__main__':
    main()
