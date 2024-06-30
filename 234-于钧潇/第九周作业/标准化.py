import numpy as np
import matplotlib.pyplot as plt

def Normalization1(x):
    return np.array([(float(i) - min(x))/float(max(x) - min(x)) for i in x])

def Normalization2(x):
    return np.array([(float(i) - np.mean(x))/float(max(x) - min(x)) for i in x])

def z_score(x):
    x_mean = np.mean(x)
    x_std = np.std(x)
    return np.array([(float(i) - x_mean)/ x_std for i in x])

if __name__ == '__main__':
    x = np.random.randint(-2, 4, size=100)
    a = Normalization1(x)
    b = Normalization2(x)
    c = z_score(x)
    plt.subplot(3, 1, 1)
    plt.plot(x, label='x')
    plt.plot(a, label='a')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(x, label='x')
    plt.plot(b, label='b')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(x, label='x')
    plt.plot(c, label='c')
    plt.legend()
    plt.show()