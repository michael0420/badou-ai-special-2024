# -*- coding:utf-8 -*-

__author__ = 'Young'

import numpy as np
import math


def gaussian_kernel(step, sigma):
    g_kernel = np.zeros([step, step])
    coef_1 = 1 / (2 * math.pi * math.pow(sigma, 2))  # 1/(2πσ²)
    coef_2 = -1 / (2 * math.pow(sigma, 2))  # -1/(2σ²)
    coor_range = [i - step // 2 for i in range(step)]  # The coordinate range of the kernel
    for i in range(step):
        for j in range(step):
            index_e = (math.pow(coor_range[i], 2) + math.pow(coor_range[j], 2)) * coef_2  # -(x²+y²)/(2σ²)
            g_kernel[i, j] = coef_1 * math.exp(index_e)  # The gaussian kernel formula
    return g_kernel / g_kernel.sum()


def sobel_kernel_x():
    return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])


def sobel_kernel_y():
    return np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
