# Author: Zhenfei Lu
# Created Date: 6/25/2024
# Version: 1.0
# Email contact: luzhenfei_2017@163.com, zhenfeil@usc.edu

from enum import Enum
import numpy as np

class LossFuncBase(object):
    def __init__(self):
        self.LossFuncType = LossFuncType.NONE

    def func(self, yPredicted, yTruth):
        pass

    def funcDerivative(self, yPredicted, yTruth):
        pass

class MSE(LossFuncBase):
    def __init__(self):
        super().__init__()
        self.LossFuncType = LossFuncType.MSE

    def func(self, yPredicted, yTruth):
        loss = np.linalg.norm(x=(yPredicted - yTruth), ord=2, axis=0)
        return loss

    def funcDerivative(self, yPredicted, yTruth):
        return yPredicted - yTruth

class CROSSENTROPY(LossFuncBase):
    def __init__(self):
        super().__init__()
        self.LossFuncType = LossFuncType.CROSSENTROPY

    def func(self, yPredicted, yTruth):
        loss = -np.sum(yTruth*np.log(yPredicted), axis=0)
        return loss

    def funcDerivative(self, yPredicted, yTruth):
        return yPredicted - yTruth

class LossFuncType(Enum):
    NONE = 0
    MSE = 1
    CROSSENTROPY = 2

