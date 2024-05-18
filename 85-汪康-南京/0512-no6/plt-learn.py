# -*- coding: utf-8 -*-
'''@Time: 2024/5/19 0:44

'''
import matplotlib.pyplot as plt
import numpy as np

# f1,plot 1:
xpoints = np.array([0, 6])
ypoints = np.array([0, 100])

plt.figure(1)
plt.subplot(1, 2, 1)
plt.plot(xpoints, ypoints)
plt.title("plot 1")

plt.suptitle("RUNOOB subplot Test")

# f2,plot 2:
x = np.array([1, 2, 3, 4])
y = np.array([1, 7, 9, 15])

plt.figure(2)
plt.subplot(1, 2, 2)
plt.plot(x, y)
plt.title("plot 2")

plt.suptitle("RUNOOB subplot Test")
plt.show()