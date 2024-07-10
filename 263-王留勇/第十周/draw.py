"""
画图
"""
import os

import numpy
import matplotlib.pyplot as plt

cur_dir = os.getcwd()
par_dir = os.path.dirname(cur_dir)
cvs_dir = os.path.join(par_dir, 'dataset/mnist_test.csv')

data_file = open(cvs_dir)
data_list = data_file.readlines()
data_file.close()
print(len(data_list))
print(data_list[0])

all_values = data_list[0].split(',')
image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
plt.imshow(image_array, cmap='Greys', interpolation='None')
plt.show()

scaled_input = image_array / 255.0 * 0.99 + 0.01
print(scaled_input)