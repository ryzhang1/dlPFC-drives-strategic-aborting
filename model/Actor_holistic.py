#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from Actor import init_weights


class Actor(nn.Module):
    def __init__(self, OBS_DIM, ACTION_DIM, TARGET_DIM, RNN_SIZE, FC_SIZE, RNN):
        super().__init__()
        self.rnn = RNN(input_size=OBS_DIM + ACTION_DIM + TARGET_DIM, hidden_size=220)
        self.l1 = nn.Linear(220, ACTION_DIM)
        
        self.apply(init_weights)
        
   
    def forward(self, x, hidden_in, return_hidden=True, critic=None): 
        t = x[..., [-1]]; x = x[..., :-1]

        if hidden_in is None:
            x, hidden_out = self.rnn(x)
        else:
            x, hidden_out = self.rnn(x, hidden_in)
         
        x = torch.tanh(self.l1(x))
        if return_hidden:
            return x, hidden_out
        else:
            return x

