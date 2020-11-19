import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.signal import convolve2d
import math
from scipy import misc
import os
from PIL import Image
import cv2

coverPath = 'C:/Users/cq/Pictures/test/'
stegoPath = 'C:/Users/cq/Pictures/testS'

def HILL_cost(cover_path):

    HF1 = np.array([[-1, 2, -1],[2, -4, 2],[-1, 2, -1]])
    W1_kernel = np.ones((3, 3), np.float32)/9 #avg filter kernel

    cover = misc.imread(cover_path, flatten=False, mode='L')
    k, l = cover.shape

    S1, S1_ = HF1.shape
    padSize = S1
    coverPadded = np.lib.pad(cover, padSize, 'symmetric')
    R1 = convolve2d(coverPadded, HF1, mode='same')
    W1 = convolve2d(np.abs(R1), W1_kernel, mode='same')
    if S1 % 2 == 0:
        W1 = np.roll(W1, [1, 0])
    if S1_ % 2 == 0:
        W1 = np.roll(W1, [0, 1])
    S_W1, S_W1_ = W1.shape
    x_ = int((S_W1-k)/2)
    y_ = int((S_W1_-l)/2)
    W1 = W1[x_ : -x_, y_ : -y_]
    S_W2, S_W2_ = W1.shape
    
    divisor = np.ones((S_W2, S_W2_))
    bias = np.add(W1, 1.0000e-10)
    rho = divisor/np.add(W1, bias)
    #misc.imsave('C:/Users/cq/Pictures/testS/testrho.png', rho)

    HW_kernel = np.ones((15, 15), np.float32)/225
    cost = cv2.filter2D(rho.astype('float32'), -1, HW_kernel, borderType=cv2.BORDER_CONSTANT)

    #misc.imsave('C:/Users/cq/Pictures/testS/test.png', cost)

for home, dirs, files in os.walk(coverPath):
    for file in files:
        if not file.startswith('.'):
            imgpath = os.path.join(home, file)
            print(imgpath)
            HILL_cost(imgpath)
            
           