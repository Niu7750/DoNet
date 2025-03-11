# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:32:24 2020

@author: 77509
"""


import numpy as np
import math as m
import matplotlib.pyplot as plt

def create_filter(fs, fl, fh, length):
    """
    The FIR band-pass filters based on window function method, where rectangular window is used

    input:
        fs: the sampling frequency (Hz)
        fl: lower-cut-off frequency (Hz)
        fh: higher-cut-off frequency (Hz)
        length: the length of the designed filter (need be set as odd)
    """

    wl = fl / fs * 2 * m.pi
    wh = fh / fs * 2 * m.pi

    middle = (length - 1) / 2
    fliter = np.zeros(length)

    for i in range(length):
        if i == middle:
            fliter[i] = (wh - wl)/m.pi
        else:
            fliter[i] = m.sin(wh * (i - middle)) / (m.pi * (i - middle)) - m.sin(wl * (i - middle)) / (
                        m.pi * (i - middle))

    return fliter



