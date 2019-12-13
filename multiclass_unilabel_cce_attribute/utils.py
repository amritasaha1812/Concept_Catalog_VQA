#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:40:11 2019

@author: amrita
"""

import torch

def one_hot(x, m, n, device):
    x_one_hot = torch.zeros((m, n)).to(device)
    x = x.unsqueeze(1)
    x_one_hot = x_one_hot.scatter_(1, x, 1)
    return x_one_hot

