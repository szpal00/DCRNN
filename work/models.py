#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F




class DBnTanh(nn.Module):
    def __init__(self,input_size, output_size, seq_len = 12):
        super(DBnTanh, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.seq_len = seq_len
        
        self.dense = nn.Sequential(
            nn.Linear(self.input_size, self.output_size),
            nn.BatchNorm1d(self.seq_len),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.dense(x)
    




class DenseModel(nn.Module):
    def __init__(self, input_size, output_size, batch_size = 256):
        super(DenseModel, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.layers = nn.Sequential(nn.Flatten(start_dim = 2),
                                    DBnTanh(self.input_size[1]*self.input_size[2],1000, seq_len=12),
                                    nn.BatchNorm1d(12),
                                    nn.Linear(1000, self.output_size)
                                   )

    
    def forward(self, x):        
        return self.layers(x)
    


class CNNModel(nn.Module):
    def __init__(self, input_size,  output_size,model, batch_size = 256):
        super(CNNModel, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.preprocess = model
        
        
        self.conv = nn.Sequential(nn.Conv1d(2*207, 512, 3, padding = 'same', stride = 1), #512, 6
                                  nn.BatchNorm1d(512),
                                  nn.ReLU(),
                                  nn.Conv1d(512, 1024, 3, padding = 'same', stride = 1), #1024,3
                                  nn.BatchNorm1d(1024),
                                  nn.ReLU(),
                                  nn.Conv1d(1024, 1024, 3, padding = 'same', stride = 1), #1024,1
                                  nn.BatchNorm1d(1024),
                                  nn.ReLU()
        
        )
        self.dense = nn.Linear(1024, self.output_size)

    
    def forward(self, x): 
        x = torch.transpose(self.preprocess(x),1,2)
        x = torch.transpose(self.conv(x),1,2)
        x = self.dense(x)
        return x
    
    
    


class LSTMModel(nn.Module):
    def __init__(self,input_size, output_size, num_directions = 1, mapping_dim = 1000, batch_size=32, device = "cuda:1"):
        super(LSTMModel, self).__init__()
        
        self.batch_size=batch_size
        
        self.input_size = input_size
        self.output_size = output_size
        self.seq_len = 12
        self.n_hidden = 50 # number of hidden states
        self.n_layers = 10 # number of LSTM layers (stacked)
        self.dir = num_directions
        self.mapping_dim = mapping_dim
        self.latent_dim = 200
    
        self.l_lstm = torch.nn.LSTM(input_size = self.input_size, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True,
                                 bidirectional = bool(num_directions-1),
                                 dropout = 0.5)
        


        
        self.dense_ = nn.Linear(self.input_size,self.mapping_dim)
        self.dense1 = DBnTanh(self.n_hidden,1000)
        self.dense2 = nn.Linear(1000,self.output_size)
        self.dropout = nn.Dropout(0.5)
        self.batchnorm1 = nn.BatchNorm1d(self.seq_len)
        
    def init_hidden(self, batch_size):
        hidden_state = torch.randn(self.dir*self.n_layers,batch_size,self.n_hidden).to(device)
        cell_state = torch.randn(self.dir*self.n_layers,batch_size,self.n_hidden).to(device)
        self.hidden = (hidden_state, cell_state)


    
    def forward(self, x):        
        batch_size, seq_len, _ = x.size()
        lstm_out, __ = self.l_lstm(x, self.hidden)
        x = lstm_out
        x = self.dropout(x)
        x = self.batchnorm1(x)
        x = self.dense1(x)
        x = self.dense2(x)
       
        
        
       
        return x
    


class LSTM_pretrained(nn.Module):
    def __init__(self,input_size, output_size, model, num_directions = 1, mapping_dim = 1000, batch_size=32, device = 0):
        super(LSTM_pretrained, self).__init__()
        
        self.batch_size=batch_size
        
        self.input_size = input_size
        self.output_size = output_size
        self.seq_len = 12
        self.n_hidden = 50 # number of hidden states
        self.n_layers = 10 # number of LSTM layers (stacked)
        self.dir = num_directions
        self.mapping_dim = mapping_dim
        
        #temporary:
        self.preprocess = model
    
        self.l_lstm = torch.nn.LSTM(input_size = self.input_size, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True,
                                 bidirectional = bool(num_directions-1),
                                 dropout = 0.5)
        


        
        self.dense_ = nn.Linear(self.input_size,self.mapping_dim)
        self.dense1 = DBnTanh(self.n_hidden,1000)
        self.dense2 = nn.Linear(self.n_hidden,self.output_size)
        self.dropout = nn.Dropout(0.5)
        self.batchnorm1 = nn.BatchNorm1d(self.seq_len)
        self.freeze_weights()
        
    def init_hidden(self, batch_size):
        hidden_state = torch.randn(self.dir*self.n_layers,batch_size,self.n_hidden).to(device)
        cell_state = torch.randn(self.dir*self.n_layers,batch_size,self.n_hidden).to(device)
        self.hidden = (hidden_state, cell_state)


    
    def forward(self, x):        
        
        x = self.preprocess(x)
        lstm_out, __ = self.l_lstm(x, self.hidden)
        x = lstm_out
        #x = self.dropout(x)
        x = self.batchnorm1(x)
        #x = self.dense1(x)
        x = self.dense2(x)
        return x
        

