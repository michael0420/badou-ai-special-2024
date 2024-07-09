import numpy as np
from Normalize import NormalizeUtils
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = np.random.uniform(0,255,(50,1))
    res1 = NormalizeUtils.NormalizedMethod1(data)
    res2 = NormalizeUtils.NormalizedMethod2(data)
    res3 = NormalizeUtils.NormalizedMethod3(data)
    plt.figure()
    plt.plot(data)
    plt.figure()
    plt.plot(res1)
    plt.plot(res2)
    plt.plot(res3)
    plt.show()