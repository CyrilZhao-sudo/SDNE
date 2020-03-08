# -*- coding: utf-8 -*-
# @Date   : 2020/3/5
# @File   : basemodel.py
# @Author : zhaochen

import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, accuracy_score




class GraphBaseModel(nn.Module):

    def __init__(self):
        super().__init__()
        pass

    def fit(self):
        pass


class TopKRanker(OneVsRestClassifier):

    def predict(self, X, top_k_list):
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return np.asarray(all_labels)

class MultiClassifier(object):
    '''
    learn from:
    https://github.com/shenweichen/GraphEmbedding/blob/master/ge/classify.py
    '''

    def __init__(self, embeddings, clf):
        self.embeddings = embeddings
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer()

    def fit(self, X, y, y_all):
        '''
        :param X:
        :param y:
        :param y_all: 所有的标签
        :return:
        '''
        self.binarizer.fit(y_all)
        X_train = [self.embeddings[x] for x in X]
        y_train = self.binarizer.transform(y)
        self.clf.fit(X_train, y_train)

    def predict(self, X, top_k_list):
        X_ = np.asarray([self.embeddings[x] for x in X])
        y_pred = self.clf.predict(X_, top_k_list=top_k_list)
        return y_pred

    def evaluate(self, X, y):
        top_k_list = [len(l) for l in y]
        y_pred = self.predict(X, top_k_list)
        y = self.binarizer.transform(y)
        averages = ["micro", "macro", "samples", "weighted"]
        results = {}
        for average in averages:
            results[average] = f1_score(y, y_pred, average=average)
        results['acc'] = accuracy_score(y, y_pred)
        print('-------------------')
        print(results)
        print('-------------------')
        return results

    def evaluate_hold_out(self, X, y, test_size=0.2, random_state=123):
        np.random.seed(random_state)
        train_size = int((1-test_size) * len(X))
        shuffle_indices = np.random.permutation(np.arange(len(X)))
        X_train = [X[shuffle_indices[i]] for i in range(train_size)]
        y_train = [y[shuffle_indices[i]] for i in range(train_size)]
        X_test = [X[shuffle_indices[i]] for i in range(train_size, len(X))]
        y_test = [y[shuffle_indices[i]] for i in range(train_size, len(X))]

        self.fit(X_train, y_train, y)

        return self.evaluate(X_test, y_test)

