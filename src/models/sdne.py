# -*- coding: utf-8 -*-
# @Date   : 2020/3/7
# @File   : sdne.py

import torch
from .basemodel import GraphBaseModel
from ..utils import process_nxgraph
import numpy as np
import scipy.sparse as sparse
from ..utils import Regularization


class SDNEModel(torch.nn.Module):

    def __init__(self, input_dim, hidden_layers, alpha, beta, device="cpu"):
        '''
        Structural Deep Network Embedding（SDNE）
        :param input_dim: 节点数量 node_size
        :param hidden_layers: AutoEncoder中间层数
        :param alpha: 对于1st_loss的系数
        :param beta: 对于2nd_loss中对非0项的惩罚
        :param device:
        '''
        super(SDNEModel, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.device = device
        input_dim_copy = input_dim
        layers = []
        for layer_dim in hidden_layers:
            layers.append(torch.nn.Linear(input_dim, layer_dim))
            layers.append(torch.nn.ReLU())
            input_dim = layer_dim
        self.encoder = torch.nn.Sequential(*layers)

        layers = []
        for layer_dim in reversed(hidden_layers[:-1]):
            layers.append(torch.nn.Linear(input_dim, layer_dim))
            layers.append(torch.nn.ReLU())
            input_dim = layer_dim
        # 最后加一层输入的维度
        layers.append(torch.nn.Linear(input_dim, input_dim_copy))
        layers.append(torch.nn.ReLU())
        self.decoder = torch.nn.Sequential(*layers)
        # torch中的只对weight进行正则真难搞啊
        # self.regularize = Regularization(self.encoder, weight_decay=gamma).to(self.device) + Regularization(self.decoder,weight_decay=gamma).to(self.device)


    def forward(self, A, L):
        '''
        输入节点的领接矩阵和拉普拉斯矩阵，主要计算方式参考论文
        :param A: adjacency_matrix, dim=(m, n)
        :param L: laplace_matrix, dim=(m, m)
        :return:
        '''
        Y = self.encoder(A)
        A_hat = self.decoder(Y)
        # loss_2nd 二阶相似度损失函数
        beta_matrix = torch.ones_like(A)
        mask = A != 0
        beta_matrix[mask] = self.beta
        loss_2nd = torch.mean(torch.sum(torch.pow((A - A_hat) * beta_matrix, 2), dim=1))
        # loss_1st 一阶相似度损失函数 论文公式(9) alpha * 2 *tr(Y^T L Y)
        loss_1st =  self.alpha * 2 * torch.trace(torch.matmul(torch.matmul(Y.transpose(0,1), L), Y))
        return loss_2nd + loss_1st




class SDNE(GraphBaseModel):

    def __init__(self, graph, hidden_layers=None, alpha=1e-5, beta=5, gamma=1e-5, device="cpu"):
        super().__init__()
        self.graph = graph
        self.idx2node, self.node2idx = process_nxgraph(graph)
        self.node_size = graph.number_of_nodes()
        self.edge_size = graph.number_of_edges()
        self.sdne = SDNEModel(self.node_size, hidden_layers, alpha, beta)
        self.device = device
        self.embeddings = {}
        self.gamma = gamma

        adjacency_matrix, laplace_matrix = self.__create_adjacency_laplace_matrix()
        self.adjacency_matrix = torch.from_numpy(adjacency_matrix.toarray()).float().to(self.device)
        self.laplace_matrix = torch.from_numpy(laplace_matrix.toarray()).float().to(self.device)

    def fit(self, batch_size=512, epochs=1, initial_epoch=0, verbose=1):
        num_samples = self.node_size
        self.sdne.to(self.device)
        optimizer = torch.optim.Adam(self.sdne.parameters())
        if self.gamma:
            regularization = Regularization(self.sdne, gamma=self.gamma)
        if batch_size >= self.node_size:
            batch_size = self.node_size
            print('batch_size({0}) > node_size({1}),set batch_size = {1}'.format(
                batch_size, self.node_size))
            for epoch in range(initial_epoch, epochs):
                loss_epoch = 0
                optimizer.zero_grad()
                loss = self.sdne(self.adjacency_matrix, self.laplace_matrix)
                if self.gamma:
                    reg_loss = regularization(self.sdne)
                    # print("reg_loss:", reg_loss.item(), reg_loss.requires_grad)
                    loss = loss + reg_loss
                loss_epoch += loss.item()
                loss.backward()
                optimizer.step()
                if verbose > 0:
                    print('Epoch {0}, loss {1} . >>> Epoch {2}/{3}'.format(epoch + 1, round(loss_epoch / num_samples, 4), epoch+1, epochs))
        else:
            steps_per_epoch = (self.node_size - 1) // batch_size + 1
            for epoch in range(initial_epoch, epochs):
                loss_epoch = 0
                for i in range(steps_per_epoch):
                    idx = np.arange(i * batch_size, min((i+1) * batch_size, self.node_size))
                    A_train = self.adjacency_matrix[idx, :]
                    L_train = self.laplace_matrix[idx][:,idx]
                    # print(A_train.shape, L_train.shape)
                    optimizer.zero_grad()
                    loss = self.sdne(A_train, L_train)
                    loss_epoch += loss.item()
                    loss.backward()
                    optimizer.step()

                if verbose > 0:
                    print('Epoch {0}, loss {1} . >>> Epoch {2}/{3}'.format(epoch + 1, round(loss_epoch / num_samples, 4),
                                                                         epoch + 1, epochs))

    def get_embeddings(self):
        if not self.embeddings:
            self.__get_embeddings()
        embeddings = self.embeddings
        return embeddings

    def __get_embeddings(self):
        embeddings = {}
        with torch.no_grad():
            self.sdne.eval()
            embed = self.sdne.encoder(self.adjacency_matrix)
            for i, embedding in enumerate(embed.numpy()):
                embeddings[self.idx2node[i]] = embedding
        self.embeddings = embeddings


    def __create_adjacency_laplace_matrix(self):
        node_size = self.node_size
        node2idx = self.node2idx
        adjacency_matrix_data = []
        adjacency_matrix_row_index = []
        adjacency_matrix_col_index = []
        for edge in self.graph.edges():
            v1, v2 = edge
            edge_weight = self.graph[v1][v2].get("weight", 1.0)
            adjacency_matrix_data.append(edge_weight)
            adjacency_matrix_row_index.append(node2idx[v1])
            adjacency_matrix_col_index.append(node2idx[v2])
        adjacency_matrix = sparse.csr_matrix((adjacency_matrix_data,
                                              (adjacency_matrix_row_index, adjacency_matrix_col_index)),
                                             shape=(node_size, node_size))
        # L = D - A  有向图的度等于出度和入度之和; 无向图的领接矩阵是对称的，没有出入度之分直接为每行之和
        # 计算度数
        adjacency_matrix_ = sparse.csr_matrix((adjacency_matrix_data+adjacency_matrix_data,
                                               (adjacency_matrix_row_index+adjacency_matrix_col_index,
                                                adjacency_matrix_col_index+adjacency_matrix_row_index)),
                                              shape=(node_size, node_size))
        degree_matrix = sparse.diags(adjacency_matrix_.sum(axis=1).flatten().tolist()[0])
        laplace_matrix = degree_matrix - adjacency_matrix_
        return adjacency_matrix, laplace_matrix


