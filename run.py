# -*- coding: utf-8 -*-
# @Date   : 2020/3/5
# @File   : run.py

from src.models.sdne import SDNE
from src.models.basemodel import MultiClassifier
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


def read_node_label(file_path, skip_head=False):
    X, y = [], []
    with open(file_path, "r") as f:
        if skip_head:
            f.readline()
        for line in f.readlines():
            tmp = line.strip().split(" ")
            X.append(tmp[0])
            y.append(tmp[1:])
    return X, y

def plot_embeddings(embeddings, X, y):
    embed_list = []
    for node in X:
        embed_list.append(embeddings[node])
    tsne = TSNE(n_components=2)
    node_tsned = tsne.fit_transform(np.asarray(embed_list), y)
    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(y[i][0], [])
        color_idx[y[i][0]].append(i)
    for c, idx in color_idx.items():
        plt.scatter(node_tsned[idx, 0], node_tsned[idx, 1], label=c)
    plt.legend()
    plt.show()

if __name__ == '__main__':

    import networkx as nx
    G = nx.read_edgelist('./data/wiki/Wiki_edgelist.txt',
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

    model = SDNE(G, hidden_layers=[256, 128])
    model.fit(batch_size=1024, epochs=20)
    embeddings = model.get_embeddings()

    X, y = read_node_label('./data/wiki/wiki_labels.txt')
    
    model = MultiClassifier(embeddings, LogisticRegression())

    model.evaluate_hold_out(X, y)

    plot_embeddings(embeddings, X, y)
