import numpy as np
import pickle
import csv
import os
import sys

import pandas as pd
import pandas.io.sql as psql

import pyodbc as db
import pdb

import torch
import torch.nn as nn
from torch import optim

import pprint
import re


"Basic implementation of GRU module using Pytorch"


class GRU_module(nn.Module):
    def __init__(self, input_size, hidden_size,output_size):
        super(GRU_module, self).__init__()
        self.hidden_size = hidden_size
        #self.gru = nn.GRU(input_size, hidden_size)#   ,batch_first=False, dropout=0.5)
        
        self.gru = nn.RNN(input_size, hidden_size)
        
        #self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(hidden_size, output_size)
  
    
    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        
        rearranged = hidden.view(hidden.size()[1], hidden.size(2))
        output = self.linear(rearranged)
        return output, hidden
    
    def initHidden(self):

        return torch.zeros(1,1, self.hidden_size)
    


def training(input_tensor, target_tensor, gru_mod, optimizer, criterion ):
    hidden_tensor = gru_mod.initHidden()
    optimizer.zero_grad()
        
    input_length = input_tensor.size
    target_length = target_tensor.size
    outputs= torch.zeros(gru_mod.hidden_size, gru_mod.hidden_size)
    
    loss=0
    pdb.set_trace()
    gru_out, gru_hid = gru_mod(input_tensor, hidden_tensor)
    #outputs[i] = gru_out
    
    temp= gru_out.view(gru_out.size(0), gru_out.size(0),target_tensor.size(0));
    temp_tar = (target_tensor.view(gru_out.size(0),gru_out.size(0),target_tensor.size(0),)).long()
    
    
    for k in range(temp.size(0)):
        pdb.set_trace()
        loss += criterion(temp[0][0][k], temp_tar[0][0][k])

        
    
    loss.backward()

    optimizer.step()
    return loss.item() / target_length
    
