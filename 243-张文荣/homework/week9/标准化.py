import numpy as np
import matplotlib.pyplot as plt

# (0~1)
def Normalization1(x):
    return [(float(i) - min(x)) / float(max(x) - min(x)) for i in x]
# (-1~1)
def Normalizayion2(x):
    return [(float(i) - np.mean(x)) / (max(x) - min(x)) for i in x]




