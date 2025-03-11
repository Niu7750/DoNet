# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 17:00:33 2020

@author: 77509
"""


from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def tensor_dataset(xnp, ynp, batch_size, a):
    xnp = xnp.astype('float32')
    ynp = ynp.squeeze()
    
    X_train = torch.from_numpy(xnp)
    X_train = X_train.unsqueeze(1)
    Y_train = torch.from_numpy(ynp).long()
    
    dataset = TensorDataset(X_train,Y_train)
    data_train = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = a)
    return data_train

    