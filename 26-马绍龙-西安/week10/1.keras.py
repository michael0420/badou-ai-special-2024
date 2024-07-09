[1]
'''
将训练数据和检测数据加载到内存中(第一次运行需要下载数据，会比较慢):
train_images是用于训练系统的手写数字图片;train_images.shape =  (60000, 28, 28)
train_labels是用于标注图片的信息;tran_labels =  [5 0 4 ... 5 6 8]
test_images是用于检测系统训练效果的图片；test_images.shape =  (10000, 28, 28)
test_labels是test_images图片对应的数字标签。test_labels [7 2 1 ... 4 5 6]

1.train_images.shape打印结果表明，train_images是一个含有60000个元素的数组.
数组中的元素是一个二维数组，二维数组的行和列都是28.即一个数字图片的大小是28*28.
2.train_lables打印结果表明，第一张手写数字图片的内容是数字5，第二张图片是数字0，以此类推.
3.test_images.shape的打印结果表示，用于检验训练效果的图片有10000张。
4.test_labels输出结果表明，用于检测的第一张图片内容是数字7，第二张是数字2，依次类推。
'''
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images.shape = ', train_images.shape)
print('tran_labels = ', train_labels)
print('test_images.shape = ', test_images.shape)
print('test_labels', test_labels)

[2]
'''
把用于测试的第一张图片打印出来
'''
test_pic1 = test_images[0]
import matplotlib.pyplot as plt

plt.imshow(test_pic1, cmap=plt.cm.binary)
plt.show()

[3]
'''
使用tensorflow.Keras搭建一个有效识别图案的神经网络，
1.models.Sequential():表示把每一个数据处理层串联起来.  (串行计算模型)
2.layers:表示神经网络中的一个数据处理层。(dense:神经元全连接)
3.layers.Dense(…):构造一个数据处理层。
4.input_shape(28*28,):表示当前处理层接收的数据格式必须是长和宽都是28的二维数组，后面的“,“表示传入的shape是二维，没有的话就是1维了，不符合参数要求
5.损失函数使用交叉熵categorical_crossentropy
'''
from tensorflow.keras import models
from tensorflow.keras import layers

network = models.Sequential()
network.add(layers.Dense(784, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                metrics=['accuracy'])

[4]
'''
在把数据输入到网络模型之前，把数据做归一化处理:
1.reshape(60000, 28*28）:train_images数组原来含有60000个元素，每个元素是一个28行，28列的二维数组，现在把每个二维数组转变为一个含有28*28个元素的一维数组.
2.由于数字图案是一个灰度图，图片中每个像素点值的大小范围在0到255之间.
3.train_images.astype(“float32”)/255 把每个像素点的值从范围0-255转变为范围在0-1之间的浮点值。
'''
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

'''
把图片对应的标记也做一个更改：目前所有图片的数字图案对应的是0到9。
例如test_images[0]对应的是数字7的手写图案，那么其对应的标记test_labels[0]的值就是7。
我们需要把数值7变成一个含有10个元素的数组，然后在第8个元素设置为1，其他元素设置为0。
例如test_lables[0] 的值由7转变为数组[0,0,0,0,0,0,0,1,0,0] ---one hot
'''
from tensorflow.keras.utils import to_categorical

print("before encoding:", test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after one hot encoding: ", test_labels[0])

[5]
'''
把数据输入网络进行训练：
train_images：用于训练的手写数字图片；
train_labels：对应的是图片的标记；
batch_size：每次网络从输入的图片数组中随机选取128个作为一组进行计算。
epochs:每次计算的循环是五次
'''
network.fit(train_images, train_labels, epochs=2, batch_size=128)
network.save('mnist_model.h5')  # 保存模型到本地

[6]
'''
测试数据输入，检验网络学习后的图片识别效果.
识别效果与硬件有关（CPU/GPU）.
'''
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print('test_loss==', test_loss)
print('test_acc==', test_acc)

[7]
'''
在测试集中选择一张手写数字到模型中，进行推理，观察识别效果
'''
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
test_pic = test_images[1234]
plt.imshow(test_pic, cmap=plt.cm.binary)
plt.show()

predictions = models.load_model('mnist_model.h5').predict(test_pic.reshape((1, 28 * 28)))  # 加载本地模型做推理
# predictions = network.predict(test_images)
print(predictions)
for i in range(10):  # 结果为10位的one hot编码，遍历结果
    if (predictions[0][i] == 1):  # predictions是一个二维数组，当前只有一个推理结果
        print("the number for the picture is : ", i)
        break
