#!/usr/bin/env python
# coding: utf-8

# In[2]:



import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import networkx as nx
from karateclub import DeepWalk, NetMF, GLEE, Node2Vec, Diff2Vec
from torchmetrics import MeanAbsolutePercentageError
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os
#from torchsummary import summary
    
def train(model, optimizer, batch_size, trainset, current_epoch, device = torch.device("cuda:1"), lstm = True):
    loss_function = nn.L1Loss()
    mse = nn.MSELoss()
    mape = MeanAbsolutePercentageError().to(device)
    
    train_loss = 0.0
    metric_scores = [0.0,0.0,0.0,0.0]
    trainset_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers = 8)
        
    model = model.train()
    for i, (x,y) in enumerate(trainset_loader):
        x = x.to(device)
        y = y.to(device)
        if lstm:
            model.init_hidden(batch_size)
        predictions = model(x)
        loss = loss_function(predictions, y)
        scores = [loss.detach().item(), mse(predictions, y).detach().item(), torch.sqrt(mse(predictions, y)).detach().item(), mape(predictions, y).detach().item()]        
        metric_scores = [metric_scores[j]+scores[j] for j in range(4)]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.detach().item()
    train_loss = train_loss / (i+1)
    for j in range(4):
        metric_scores[j] = metric_scores[j]/(i+1)
    
    
    return model

def evaluate(model, optimizer, batch_size, testset,current_epoch, device = torch.device("cuda:1"), lstm = True, test = False):
    loss_function = nn.L1Loss(reduction='mean')
    mse = nn.MSELoss()
    mape = MeanAbsolutePercentageError().to(device)
    test_loss = 0.0
    metric_scores = [0.0,0.0,0.0,0.0]
    testset_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers = 8)
        
    model = model.eval()
    for i, (x,y) in enumerate(testset_loader):
        x = x.to(device)
        
        if testset.scaling:
            tmp, tmp2 = torch.split(y, 207, 2)
            scaled_tmp = testset.scaler.inverse_transform(tmp)
            y=torch.cat((scaled_tmp, tmp2), 2)
        y = y.to(device)
        if lstm:
            model.init_hidden(batch_size)
        predictions = model(x)
        if testset.scaling:
            tmp, tmp2 = torch.split(predictions, 207, 2)
            scaled_tmp = testset.scaler.inverse_transform(tmp)
            predictions=torch.cat((scaled_tmp, tmp2), 2)
        predictions = model(x)
        loss = loss_function(predictions, y)
        scores = [loss.detach().item(), mse(predictions, y).detach().item(), torch.sqrt(mse(predictions, y)).detach().item(), mape(predictions, y).detach().item()]
        metric_scores = [metric_scores[j]+scores[j] for j in range(4)]
        test_loss += loss.detach().item()
    test_loss = test_loss / (i+1)
    for j in range(4):
        metric_scores[j] = metric_scores[j]/(i+1)
    if test:
        print(f"Epoch: {current_epoch} | Test MAE: {test_loss} | Test MSE {metric_scores[1]} | Test RMSE {metric_scores[2]} | Test MAPE {metric_scores[3]}")
    
    return metric_scores
    

