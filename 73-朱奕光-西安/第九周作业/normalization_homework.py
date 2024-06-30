import numpy as np
import matplotlib.pyplot as plt

def Normalization1(x):
    data = np.array([])
    for i in x:
        s = (i - min(x)) / (max(x) - min(x))
        data = np.append(data,s)
    return data

def Normalization2(x):
    data = np.array([])
    x_mean = np.mean(x)
    for i in x:
        s = (i - x_mean)/(max(x) - min(x))
        data = np.append(data,s)
    return data

def Normalization3(x):
    data1 = np.array([])
    data2 = np.array([])
    x_mean = np.mean(x)
    for i in x:
        s1 = (i - x_mean)*(i - x_mean)
        data1 = np.append(data1,s1)
    index = sum(data1)/len(data1)
    for i in x:
        s2 = (i - x_mean)/index
        data2 = np.append(data2,s2)
    return data2

if __name__ == '__main__':
    x = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
    print(Normalization1(x))
    print(Normalization2(x))
    print(Normalization3(x))
    cs = []
    for i in x:
        c = x.count(i)
        cs.append(c)
    print(cs)
    plt.plot(Normalization1(x),cs)
    plt.plot(Normalization2(x),cs)
    plt.plot(Normalization3(x),cs)
    plt.show()