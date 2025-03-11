# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 13:03:30 2020

@author: 77509
"""


import numpy as np
import scipy.io as scio
from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws, read_raw_edf, read_raw_gdf


def divide_samples(subject, kf, start_time=0, window_length=1):
    end_time = start_time+window_length-1/250

    ''' Please modify dataset address before running '''
    filename = '/home/kangjh/kangjh_data/BCIC/BCIC_IV_2a/A0%dT.gdf' % subject

    raw = read_raw_edf(filename, preload=True)
    raw.filter(1, 48., fir_design='firwin', skip_by_annotation='edge') # 1-48Hz bandpass filtering
    event_id = dict(left=0, right=1, feet=2, tongue=3)
    events, _ = events_from_annotations(raw, event_id={'769': 0, '770': 1, '771': 2, '772': 3})
    epochs = Epochs(raw, events, event_id, start_time, end_time, proj=True, baseline=None, preload=True)
    data = epochs._data[:, :22] * (10 ** 5)

    lei0 = [i for i in range(288) if events[i, 2] == 0]
    lei1 = [i for i in range(288) if events[i, 2] == 1]
    lei2 = [i for i in range(288) if events[i, 2] == 2]
    lei3 = [i for i in range(288) if events[i, 2] == 3]

    k_size = 18
    test_x = data[lei0[kf * k_size:(kf + 1) * k_size] + lei1[kf * k_size:(kf + 1) * k_size] + lei2[kf * k_size:(kf + 1) * k_size] + lei3[kf * k_size:(kf + 1) * k_size]]
    test_y = np.array([0, 1, 2, 3]).repeat(k_size)
    lei0 = [i for i in lei0 if not i in lei0[kf * k_size:(kf + 1) * k_size]]
    lei1 = [i for i in lei1 if not i in lei1[kf * k_size:(kf + 1) * k_size]]
    lei2 = [i for i in lei2 if not i in lei2[kf * k_size:(kf + 1) * k_size]]
    lei3 = [i for i in lei3 if not i in lei3[kf * k_size:(kf + 1) * k_size]]

    train_x = data[lei0 + lei1 + lei2 + lei3]
    train_y = np.array([0, 1, 2, 3]).repeat(72 - k_size)

    return train_x, train_y, test_x, test_y
