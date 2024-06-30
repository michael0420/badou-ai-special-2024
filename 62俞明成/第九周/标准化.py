import random
import torch
import numpy as np
from matplotlib.pyplot import plot, show
import matplotlib

matplotlib.use("TKAgg")

data = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,
        11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]

cs = []
for i in data:
    c = data.count(i)
    cs.append(c)


def func1(x):
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / std


def func2(x):
    min = np.min(x)
    max = np.max(x)
    return (x - min) / (max - min)

def func3(x):
    min = np.min(x)
    max = np.max(x)
    return (x - np.mean(x)) / (max - min)


standardized_data = func1(np.array(data))
standardized_data2 = func2(np.array(data))
standardized_data3 = func3(np.array(data))
print(standardized_data)
plot(standardized_data, cs)
plot(standardized_data2, cs)
plot(standardized_data3, cs)
plot(data, cs)
show()
