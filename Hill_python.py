import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.signal import convolve2d
import math
import os
from PIL import Image
import cv2

coverPath = 'C:/Users/cq/Pictures/test/'
stegoPath = 'C:/Users/cq/Pictures/testS'


def HILL_cost(cover_path):

    HF1 = np.array([[-1., 2., -1.],[2., -4., 2.],[-1., 2., -1.]])
    W1_kernel = np.ones((3, 3), np.float16)/9. #avg filter kernel
    
    cover = cv2.imread(cover_path, 0)
    cover = (cover/255.).astype('float16')
    k, l = cover.shape

    S1, S1_ = HF1.shape
    padSize = max(S1, S1_)
    coverPadded = np.lib.pad(cover, padSize, 'symmetric')
    R1 = convolve2d(coverPadded, HF1, mode='same')
    cv2.imwrite('C:/Users/cq/Pictures/testS/testR1.png', R1)
    W1 = convolve2d(np.abs(R1), W1_kernel, mode='same')
    cv2.imwrite('C:/Users/cq/Pictures/testS/testW1.png', W1)


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
    bias = np.add(W1, 10e-10)
    rho = divisor/np.add(W1, bias)
    #cv2.imwrite('C:/Users/cq/Pictures/testS/testrho.png', rho)

    HW_kernel = np.ones((15, 15), np.float)/225
    cost = cv2.filter2D(rho.astype('float'), -1, HW_kernel, borderType=cv2.BORDER_CONSTANT)

    cv2.imwrite('C:/Users/cq/Pictures/testS/test.png', cost)

for home, dirs, files in os.walk(coverPath):
    for file in files:
        if not file.startswith('.'):
            imgpath = os.path.join(home, file)
            print(imgpath)
            HILL_cost(imgpath)
            
           
