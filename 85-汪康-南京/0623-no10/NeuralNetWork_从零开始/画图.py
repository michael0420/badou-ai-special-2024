import numpy
import matplotlib.pyplot as plt
#%matplotlib inline

#open函数里的路径根据数据存储的路径来设定
data_file = open("dataset/mnist_test.csv")
data_list = data_file.readlines()
data_file.close()
print(len(data_list))
print(data_list[0])

#把数据依靠','区分，并分别读入
all_values = data_list[0].split(',')
#第一个值对应的是图片的表示的数字，所以我们读取图片数据时要去掉第一个数值
image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
plt.imshow(image_array, cmap='Greys', interpolation='None')
plt.show()

#数据预处理（归一化）
scaled_input = image_array / 255.0 * 0.99 + 0.01
print(scaled_input)

