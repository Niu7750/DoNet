# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 13:08:19 2020

@author: 77509
"""

import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.functional import elu,sigmoid
import torch.nn.functional as F
from FIR_Filter import create_filter

class SONet(nn.Module):
    def __init__(self,
                input_channel,
                n_class,

                n_spatial =16,

                fir_bank=[],
                fir_length=51,
                batch_norm_alpha=0.1):
        super(SONet, self).__init__()
        self.__dict__.update(locals())
        del self.self

        ''' Structure '''
        self.SpatialConv= nn.Conv2d(1, self.n_spatial, (self.input_channel, 1), stride=(1, 1), bias=False, )

        self.n_fir = len(self.fir_bank)
        self.FIRConv = nn.Conv2d(1, self.n_fir, (1, self.fir_length), stride=(1, 1), bias=False, )
        self.FIRConv.weight.requires_grad = False

        self.n_PLP= self.n_fir * self.n_spatial//2
        self.DepthwiseConv = nn.Conv2d(self.n_PLP, 2*self.n_PLP, kernel_size=(2, 1), stride=(2, 1), bias=False, groups=self.n_PLP)
        self.list_even = list(np.arange(self.n_PLP) * 2)
        self.list_odd = list(np.arange(self.n_PLP) * 2 + 1)

        self.ClassConv = nn.Conv2d(1, self.n_class, kernel_size=(self.n_PLP, 1), stride=(1, 1), bias=False, )

        self.BN1 = nn.BatchNorm2d(self.n_fir, momentum=self.batch_norm_alpha, affine=True, eps=1e-5, )
        self.BN2 = nn.BatchNorm2d(1, momentum=self.batch_norm_alpha, affine=True, eps=1e-5, )
        self.Softmax = nn.LogSoftmax(dim=1)

        ''' Initialization '''
        nn.init.xavier_uniform_(self.SpatialConv.weight, gain=1)
        nn.init.xavier_uniform_(self.DepthwiseConv.weight, gain=1)
        nn.init.xavier_uniform_(self.ClassConv.weight, gain=1)
        nn.init.constant_(self.BN1.weight, 1)
        nn.init.constant_(self.BN1.bias, 0)
        nn.init.constant_(self.BN2.weight, 1)
        nn.init.constant_(self.BN2.bias, 0)

        ''' Initialized as FIR filters '''
        fir_filters = np.zeros([self.n_fir, 1, 1, self.fir_length])
        for i, lh in enumerate(self.fir_bank):
            fir_filters[i, 0, 0, :] = create_filter(fs=250, fl=lh[0], fh=lh[1], length=self.fir_length)
        time_tensor = torch.Tensor(fir_filters)
        self.FIRConv.weight = torch.nn.Parameter(time_tensor)


    def forward(self, x): # size (b, 1, 22, length)
        ''' Sub-band component extraction '''
        x = self.SpatialConv(x)
        x = x.permute(0, 2, 1, 3)
        sub_component = self.FIRConv(x)
        x = self.BN1(sub_component)

        ''' Pairing '''
        x = x.view(x.data.size(0), self.n_fir * self.n_spatial//2, 2, -1)

        ''' Transcoding '''
        x = self.DepthwiseConv(x)
        x = torch.mul(x,x)
        x = torch.sqrt(x[:,self.list_odd]+x[:, self.list_even])
        x = self.BN2(x.permute(0,2,1,3))

        ''' Classification '''
        x = self.Softmax(self.ClassConv(x))
        x = x.squeeze()

        return x, sub_component


class DoNet(nn.Module):
    def __init__(self,
                 input_channel,
                 n_class,

                 n_spatial=16,

                 fir_bank=[],
                 fir_length=51,
                 sample_frequency=250,

                 delta_n = 100,

                 batch_norm_alpha=0.1):
        super(DoNet, self).__init__()
        self.__dict__.update(locals())
        del self.self

        ''' Structure '''
        self.SpatialConv = nn.Conv2d(1, self.n_spatial, (self.input_channel, 1), stride=(1, 1), bias=False, )
        self.n_fir = len(self.fir_bank)
        self.FIRConv = nn.Conv2d(1, self.n_fir, (1, self.fir_length), stride=(1, 1), bias=False, )
        self.FIRConv.weight.requires_grad = False
        self.GroupConv = nn.Conv2d(self.n_fir, 2 * self.n_fir, kernel_size=(1, 2), stride=(1, 1),
                                   dilation=(1, self.delta_n), bias=False, groups=self.n_fir)
        self.ClassConv = nn.Conv2d(self.n_spatial, self.n_class, kernel_size=(self.n_fir, 1), stride=(1, 1),
                                     bias=False, )

        self.BN1 = nn.BatchNorm2d(self.n_fir, momentum=self.batch_norm_alpha, affine=True, eps=1e-5, )
        self.BN2 = nn.BatchNorm2d(self.n_spatial, momentum=self.batch_norm_alpha, affine=True, eps=1e-5, )
        self.Softmax = nn.LogSoftmax(dim=1)

        self.list_even = list(np.arange(self.n_fir) * 2)
        self.list_odd = list(np.arange(self.n_fir) * 2 + 1)

        ''' Initialization '''
        nn.init.xavier_uniform_(self.SpatialConv.weight, gain=1)
        nn.init.xavier_uniform_(self.GroupConv.weight, gain=1)
        nn.init.xavier_uniform_(self.ClassConv.weight, gain=1)
        nn.init.constant_(self.BN1.weight, 1)
        nn.init.constant_(self.BN1.bias, 0)
        nn.init.constant_(self.BN2.weight, 1)
        nn.init.constant_(self.BN2.bias, 0)

        ''' Initialized as FIR filters '''
        fir_filters = np.zeros([self.n_fir, 1, 1, self.fir_length])
        for i, lh in enumerate(self.fir_bank):
            fir_filters[i, 0, 0, :] = create_filter(fs=250, fl=lh[0], fh=lh[1], length=self.fir_length)
        time_tensor = torch.Tensor(fir_filters)
        self.FIRConv.weight = torch.nn.Parameter(time_tensor)


    def forward(self, x): # size (b, 1, 22, 1250)
        x = self.SpatialConv(x)
        x = x.permute(0, 2, 1, 3)
        x_sub = self.FIRConv(x)  # sub-band component
        x = self.BN1(x_sub)
        x = self.GroupConv(x)
        x = torch.mul(x,x)
        x_am = torch.sqrt(x[:,self.list_odd]+x[:, self.list_even])  # amplitude
        x = self.BN2(x_am.permute(0,2,1,3))
        x = self.Softmax(self.ClassConv(x))
        x = x.squeeze(2)
        return  x, x_sub, x_am