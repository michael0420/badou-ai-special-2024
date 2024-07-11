import numpy as np

class NormalizeUtils(object):
    def __init__(self):
        pass

    @staticmethod
    def NormalizedMethod1(x:np.ndarray):
        min = np.min(x)
        max = np.max(x)
        return (x-min)/(max-min)

    @staticmethod
    def NormalizedMethod2(x: np.ndarray):
        min = np.min(x)
        max = np.max(x)
        mean = np.mean(x)
        return (x - mean) / (max - min)

    @staticmethod
    def NormalizedMethod3(x: np.ndarray):
        # min = np.min(x)
        max = np.max(x)
        # mean = np.mean(x)
        return x / max