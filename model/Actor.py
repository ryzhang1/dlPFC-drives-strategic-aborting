#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m, mean=0, std=0.1, bias=0):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean, std)
        nn.init.constant_(m.bias, bias)
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, bias)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
            else:
                raise ValueError()


class Actor(nn.Module):
    def __init__(self, OBS_DIM, ACTION_DIM, TARGET_DIM, RNN_SIZE, FC_SIZE, RNN):
        super().__init__()
        self.OBS_DIM = OBS_DIM
        self.ACTION_DIM = ACTION_DIM
        self.value_noise_std = 0
        self.reverse_value = False
        self.reverse_t_thre = 1
       
        self.l1 = nn.Linear(RNN_SIZE + 1, FC_SIZE)
        self.l2 = nn.Linear(FC_SIZE, FC_SIZE)
        self.l3 = nn.Linear(FC_SIZE, ACTION_DIM)
        
        self.apply(init_weights)
        
        
    def get_valuediff(self, b, critic):
        with torch.no_grad():
            u1 = torch.zeros(b.shape[0], b.shape[1], 2, device=b.device)
            u2 = torch.zeros(b.shape[0], b.shape[1], 2, device=b.device); u2[..., 0] = 1

            v_ = []
            for u in [u1, u2]:
                v = F.relu(critic.l1(torch.cat([b, u], dim=2)))
                v = F.relu(critic.l2(v))
                v = critic.l3(v)
                v_.append(v)

            v = v_[0] - v_[1]
            v += v * self.value_noise_std * torch.randn_like(v)
        
        return v / 5
        
   
    def forward(self, x, hidden_in, return_hidden=True, critic=None): 
        t = x[..., [-1]]; x = x[..., :-1]

        with torch.no_grad():
            if hidden_in is None:
                b, hidden_out = critic.rnn1(x)
            else:
                b, hidden_out = critic.rnn1(x, hidden_in)
         
            v = self.get_valuediff(b, critic)
            if self.reverse_value:
                v = (t <= self.reverse_t_thre) * (-v) + v * (t > self.reverse_t_thre)
                 
        x = torch.cat([b, v], axis=-1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        if return_hidden:
            return x, hidden_out
        else:
            return x

