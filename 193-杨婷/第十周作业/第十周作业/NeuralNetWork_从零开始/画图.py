import numpy as np
import matplotlib.pyplot as plt
data_file = open("dataset/mnist_train.csv")
data_list = data_file.readlines()
data_file.close()
print(len(data_list))
print(data_list[0])

all_values = data_list[0].split(',')
# 第一个值对应的是图片的表示的数字，所以我们读取图片数据时要去掉第一个数值
image_array = np.asfarray(all_values[1:]).reshape(28, 28)  # numpy.asfarray函数尝试将输入转换为浮点数数组
plt.imshow(image_array, cmap='Greys', interpolation='None')  # interpolation插值
plt.show()
