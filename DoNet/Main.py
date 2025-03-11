# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 14:06:04 2020

@author: 77509
"""
from Crop_Trial import divide_samples
from Model import DoNet
from Tensor_Dataset import tensor_dataset

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.fftpack import fft
import time
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assume that we are on a CUDA machine, then this should print a CUDA device:
print(device)

''' Parameters of the dataset'''
n_class = 4
n_channel = 22
sampling_frequency = 250

fir_bank = []
for i in range(10,140,5):
    fir_bank.append([i/4,i/4+1.25])

''' Adjustable parameters'''
start_time=0.0
window_length=1.0
n_spatial = 16
fir_length = 51
delta_n = 100 # 1 or a suitable value
n_point = int(window_length*sampling_frequency - fir_length + 1)-delta_n # number of time points for voting

chart = np.zeros([9,4,10]) #used for recording accuracy
for subject in range(1,10):
    for k_fold in range(4): # 4-fold cross-validation
        train_x, train_y, test_x, test_y = divide_samples(subject, k_fold, start_time, window_length)
        batch_size_train = 16
        batch_size_test = 72

        test = tensor_dataset(test_x, test_y, batch_size_test, False)
        for n_run in range(10):
            train = tensor_dataset(train_x, train_y, batch_size_train, True)

            model = DoNet(input_channel = n_channel,
                           n_class = n_class,
                           n_spatial = n_spatial,
                           fir_bank=fir_bank,
                           fir_length = fir_length,
                           sample_frequency=sampling_frequency,
                           delta_n=delta_n,)
            model = model.to(device)

            criterion = nn.CrossEntropyLoss()
            model.FIRConv.weight.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, betas=(0.9, 0.999),
                                   eps=1e-08, weight_decay=1e-4)

            ACC_point =[]
            ACC_vote = []
            time_start = time.time()

            for epoch in range(800):
                running_loss=0.0
                correct = 0
                total = 0

                for i,data in enumerate(train):
                    x, y = data
                    x, y = x.to(device), y.to(device)

                    out, _, _ = model(x)

                    ''' Flatten for calculateing a loss at each time point'''
                    out_c = out[:,:,0]
                    y_c = y
                    for j in range(1,n_point):
                        out_c = torch.cat((out_c, out[:,:,j]), 0)
                        y_c = torch.cat((y_c, y))

                    loss = criterion(out_c, y_c)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    _, predicted = torch.max(out_c.data, 1)
                    correct += (predicted == y_c).sum().item()
                    total += y_c.size(0)
                    running_loss += loss.item()

                    pack = 5
                    if i % pack == pack-1: # print every 2000 mini-batches
                        acc_train = 100 * correct / total
                        print('[%d, %5d] loss: %.7f acc: %.2f %%' % (epoch + 1, i + 1, running_loss / pack, acc_train))
                        running_loss = 0.0
                        correct = 0
                        total = 0

                correct = 0
                total = 0
                total_sample = 0
                n_vote = 0
                with torch.no_grad():
                    for i,datat in enumerate(test):
                        x, y_test = datat
                        x, y_test = x.to(device), y_test.to(device)

                        out, _, _ = model(x)

                        out_c = out[:,:,0]
                        y_c = y_test
                        for j in range(1,n_point):
                            out_c = torch.cat((out_c, out[:,:,j]), 0)
                            y_c = torch.cat((y_c, y_test))

                        _, predicted = torch.max(out_c.data, 1)
                        total += y_c.size(0)
                        correct += (predicted == y_c).sum().item()

                        ''' Predict a label for each time point and vote '''
                        _, predicted = torch.max(out.data, 1)
                        predicted = predicted.cpu().numpy()
                        y_np = y_test.cpu().numpy()
                        for j in range(batch_size_test):
                            count = np.bincount(predicted[j, :])
                            p_point = np.argmax(count)
                            if p_point == y_np[j]:
                                n_vote += 1
                        total_sample += y_test.size(0)

                    acc_point = correct / total
                    acc_vote = n_vote/total_sample

                    ACC_point.append(acc_point)
                    ACC_vote.append(acc_vote)

                    print('Accuracy of the test: %.4f %%    Accuracy after vote%.4f %% ' % (100 * acc_point, 100*acc_vote))

                if acc_vote >= 1:
                    break

            time_end = time.time()
            time_c = time_end - time_start

            acc_point_max = np.max(ACC_point)

            acc_vote_max = np.max(ACC_vote) # the highest accuracy
            chart[subject,k_fold,n_run] = acc_vote_max
            acc_epoch = np.argmax(ACC_vote)
            print(acc_vote_max, acc_epoch, time_c)

            path1 = 'Result/'
            path2 = 'S%d/%d %.4f-%.4f-%d-%f(%f)'%(subject, k_fold, acc_point_max, acc_vote_max, acc_epoch, time_c, acc_vote)
            path = path1+path2
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(model, path+'/model.pth')
            np.save(path1+'acc_chart.npy', chart)


