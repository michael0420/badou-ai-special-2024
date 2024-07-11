import numpy

from NeuralNetWork import NeuralNetWork

inodes = 784
hnodes = 200
onodes = 10
lr = 0.1
n = NeuralNetWork(inodes, hnodes, onodes, lr)

#读入训练数据
#open函数里的路径根据数据存储的路径来设定
training_data_file = open("dataset/mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()


epoch = 10

for i in range(epoch):
    #把数据依靠','区分，并分别读入
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
        #设置图片与数值的对应关系
        targets = numpy.zeros(onodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

test_data_file = open("dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
scores = []

for record in test_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:])) / 255 * 0.99 + 0.01
    correct = int(all_values[0])
    print("测试图像实际数字 = ", correct)
    outputs = n.query(inputs)
    #找到数值最大的神经元的编号
    label = numpy.argmax(outputs)
    print("经过神经网络推理后的数字 = ", label)
    if label == correct:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

#计算图片的成功率
scores_array = numpy.asarray(scores)
print("perfermance = ", scores_array.sum() / scores_array.size)


#测试图片
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("dataset/my_own_2.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_arr = numpy.array(img)

print(img_arr)

print(img_arr.shape)

img_data = img_arr.flatten()

img_data_query = numpy.zeros(len(img_data), img_data.dtype)
print(img_data_query.shape)
for i in range(len(img_data)):
    if img_data[i] == 255:
        img_data_query[i] = 0
    else:
        img_data_query[i] = 1


image_array = numpy.asfarray(img_data_query).reshape((28, 28))
plt.imshow(image_array, cmap='Greys', interpolation='None')
plt.show()

print(img_data_query)

inputs = (numpy.asfarray(img_data_query)) / 255 * 0.99 + 0.01
print(inputs)

outputs = n.query(inputs)
print(outputs)
label = numpy.argmax(outputs)
print("经过神经网络推理后的数字 = ", label)