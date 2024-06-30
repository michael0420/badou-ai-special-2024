# Author: Zhenfei Lu
# Created Date: 4/15/2024
# Version: 1.0
# Email contact: luzhenfei_2017@163.com, zhenfeil@usc.edu

import numpy as np
from enum import Enum
import random

class NoiseType(Enum):
    NONE = 0
    GAUSSIAN = 1
    PEPPERSALT = 2
    POISSION = 3

class Noise(object):
    def __init__(self):
        self.type = NoiseType.NONE
        self.percentage = 0

    def addNoise4Img(self):
        print('Noise Base')
        pass

class GaussianNoise(Noise):
    def __init__(self, percentage, mu, sigma):
        super().__init__()
        self.type = NoiseType.GAUSSIAN
        self.percentage = percentage
        self.mu = mu
        self.sigma = sigma

    def addNoise4Img(self, img: np.ndarray) -> np.ndarray:
        (h, w) = img.shape[0:2]
        outputImg = img.copy() # deep(value) copy
        N = int(h*w*self.percentage)
        for i in range(0, N):
            row = random.randint(0, h-1)
            col = random.randint(0, w-1)
            noise = random.gauss(self.mu, self.sigma)
            outputImg[row, col] = outputImg[row, col] + noise
            if(outputImg[row, col] < 0):
                outputImg[row, col] = 0
            if(outputImg[row, col] > 255):
                outputImg[row, col] = 255
        return outputImg

class PepperSaltNoise(Noise):
    def __init__(self, percentage):
        super().__init__()
        self.type = NoiseType.PEPPERSALT
        self.percentage = percentage

    def addNoise4Img(self, img: np.ndarray) -> np.ndarray:
        (h, w) = img.shape[0:2]
        outputImg = img.copy() # deep(value) copy
        N = int(h*w*self.percentage)
        for i in range(0, N):
            row = random.randint(0, h-1)
            col = random.randint(0, w-1)
            if(outputImg[row, col] >= 0.5):
                outputImg[row, col] = 255
            else:
                outputImg[row, col] = 0
        return outputImg

class PoissonNoise(Noise):
    def __init__(self, percentage, lamb):
        super().__init__()
        self.type = NoiseType.POISSION
        self.percentage = percentage
        self.lamb = lamb

    def addNoise4Img(self, img: np.ndarray) -> np.ndarray:
        (h, w) = img.shape[0:2]
        outputImg = img.copy() # deep(value) copy
        N = int(h*w*self.percentage)
        for i in range(0, N):
            row = random.randint(0, h-1)
            col = random.randint(0, w-1)
            noise = np.random.poisson(self.lamb, (1,))
            outputImg[row, col] = outputImg[row, col] + noise
            if(outputImg[row, col] < 0):
                outputImg[row, col] = 0
            if(outputImg[row, col] > 255):
                outputImg[row, col] = 255
        return outputImg

