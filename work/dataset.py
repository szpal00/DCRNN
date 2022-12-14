#!/usr/bin/env python
# coding: utf-8



import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import networkx as nx
from karateclub import DeepWalk, NetMF, GLEE
from torchmetrics import MeanAbsolutePercentageError
import matplotlib.pyplot as plt
import os


class DenseDataset(Dataset):
    def __init__(self, data, adj_mtx = None, embedding = None, scaling = True):
        self.data = data
        self.adj_mtx = adj_mtx
        self.embedding = embedding
        self.scaling = scaling
        
        self.scaler = StandardScaler(mean=data['x'][..., 0].mean(), std=data['x'][..., 0].std())
        
        self.data_x = self.reshape_data('x')
        self.data_y = self.reshape_data('y')
        self.graph = None
        if self.embedding != None:
            self.init_embedding(self.embedding)
            
        
    def __len__(self):
        return len(self.data_x)
    
    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = torch.flatten(self.data_y[idx])
        x = self.add_adj_mtx(x)
        return x, y
    
    def add_adj_mtx(self, x):
        #if self.adj_mtx != None:
        #    v = torch.from_numpy(self.adj_mtx)
        #    x = torch.cat((x, v), 1)
        #    y = torch.cat((y, v), 1)
        if self.embedding != None:
            
            x = torch.cat((x, self.graph), 0)
            #y = torch.cat((y, self.graph), 0)
        return x#,y
    
    #def reshape_data(self, pos):
    #    tmp_arr = np.reshape(self.data[pos],(-1,self.data[pos].shape[2],self.data[pos].shape[3]))
    #    return F.normalize(torch.from_numpy(tmp_arr).transpose(1,2).float())
    def reshape_data(self, pos):
        tmp_arr = np.reshape(self.data[pos],(-1,self.data[pos].shape[2],self.data[pos].shape[3]))
        if self.scaling:
            scaled_tensor = torch.from_numpy(self.scaler.transform(tmp_arr[...,0]))
        else:
            scaled_tensor = torch.from_numpy(tmp_arr[...,0])
        tmp = torch.from_numpy(tmp_arr[...,1])
        shapes_1 = [i for i in scaled_tensor.shape]
        shapes_2 = [i for i in tmp.shape]
        scaled_tensor = scaled_tensor.view(shapes_1[0], shapes_1[1], 1)
        tmp = tmp.view(shapes_2[0], shapes_2[1], 1)
        ret = torch.cat((scaled_tensor, tmp), 2)
        return ret.transpose(1,2).float()
    def init_embedding(self, embedding):
        self.embedding = embedding
        Gcc = sorted(nx.connected_components(self.adj_mtx), key=len, reverse=True)
        giant = adj_mtx.subgraph(Gcc[0])
        giant = nx.convert_node_labels_to_integers(giant, first_label=0, ordering='default')
        self.embedding.fit(giant)
        vec = self.embedding.get_embedding()
        embedding_dim = len(vec[0])
        num_nodes = len(self.adj_mtx.nodes)
        tmp = np.zeros((num_nodes,embedding_dim))
        avg_vec = np.average(tmp, 0)
        j = 0
        for i in range(num_nodes):
            if i in list(Gcc[0]):
                tmp[i] = vec[j]
                j += 1
            else:
                tmp[i] = avg_vec
        
        self.graph = torch.from_numpy(tmp).transpose(0,1)
        
    def set_embedding(self, embedding):
        if embedding == None:
            self.embedding = None
        else:
            self.init_embedding(embedding)
        
    def set_scale(self, scaling):
        #method to change scaling, compute the whole dataset again if it's called
        self.scaling = scaling
        self.data_x = self.reshape_data('x')
        self.data_y = self.reshape_data('y')


# In[3]:


class LSTMDataset(Dataset):
    def __init__(self, data, adj_mtx = None, embedding = None, scaling = True):
        self.data = data
        self.adj_mtx = adj_mtx
        self.embedding = embedding
        self.scaling = scaling
        
        self.scaler = StandardScaler(mean=data['x'][..., 0].mean(), std=data['x'][..., 0].std())
        self.data_x = self.reshape_data('x')
        self.data_y = self.reshape_data('y')
        self.graph = None
        if self.embedding != None:
            self.init_embedding(self.embedding)
            
        
    def __len__(self):
        return len(self.data_x)
    
    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]
        x = self.add_adj_mtx(x)
        #x = torch.reshape(x, (x.shape[0], -1))
        y = torch.reshape(y, (y.shape[0], -1))
        return x, y
    
    def add_adj_mtx(self, x):
        #if self.adj_mtx != None:
        #    v = torch.from_numpy(self.adj_mtx)
        #    x = torch.cat((x, v), 1)
        #    y = torch.cat((y, v), 1)
        if self.embedding != None:
            
            x = torch.cat((x, self.graph), 1)
            #y = torch.cat((y, self.graph), 0)
        return x#,y
    
    def reshape_data(self, pos):
        #tmp_arr = np.reshape(self.data[pos],(-1,self.data[pos].shape[2],self.data[pos].shape[3]))
        if self.scaling:
            scaled_tensor = torch.from_numpy(self.scaler.transform(self.data[pos][...,0]))
        else:
            scaled_tensor = torch.from_numpy(self.data[pos][...,0])
        tmp = torch.from_numpy(self.data[pos][...,1])
        shapes_1 = [i for i in scaled_tensor.shape]
        shapes_2 = [i for i in tmp.shape]
        scaled_tensor = scaled_tensor.view(shapes_1[0], shapes_1[1], shapes_1[2], 1)
        tmp = tmp.view(shapes_2[0], shapes_2[1], shapes_2[2], 1)
        ret = torch.cat((scaled_tensor, tmp), 3)
        return ret.transpose(2,3).float()
    
    def init_embedding(self, embedding):
        self.embedding = embedding
        Gcc = sorted(nx.connected_components(self.adj_mtx), key=len, reverse=True)
        giant = self.adj_mtx.subgraph(Gcc[0])
        giant = nx.convert_node_labels_to_integers(giant, first_label=0, ordering='default')
        self.embedding.fit(giant)
        vec = self.embedding.get_embedding()
        embedding_dim = len(vec[0])
        num_nodes = len(self.adj_mtx.nodes)
        tmp = np.zeros((num_nodes,embedding_dim))
        avg_vec = np.average(tmp, 0)
        j = 0
        for i in range(num_nodes):
            if i in list(Gcc[0]):
                tmp[i] = vec[j]
                j += 1
            else:
                tmp[i] = avg_vec
        self.graph = tmp
        self.graph = torch.from_numpy(np.reshape(self.graph, (1, self.graph.shape[0], self.graph.shape[1]))).transpose(1,2).float()
        self.graph = self.graph.repeat(12,1, 1)
        
    def set_embedding(self, embedding):
        if embedding == None:
            self.embedding = None
        else:
            self.init_embedding(embedding)
        
    def set_scale(self, scaling):
        #method to change scaling, compute the whole dataset again if it's called
        self.scaling = scaling
        self.data_x = self.reshape_data('x')
        self.data_y = self.reshape_data('y')
            


# In[4]:


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


# In[ ]:




