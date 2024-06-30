# load data
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# (60000, 28, 28)
print('train_images.shape = ', train_images.shape)
# [5 0 4 ... 5 6 8]
print('train_labels = ', train_labels)
# (10000, 28, 28)
print('test_images.shape = ', test_images.shape)
# [7 2 1 ... 4 5 6]
print('test_labels = ', test_labels)

# show first picture
digit = test_images[0]
import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

from tensorflow.keras import models
from tensorflow.keras import layers

# 表示把每一个数据处理层串联起来.
network = models.Sequential()
# Dense:全连接层
# layers.Dense(…):构造一个数据处理层
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#转为一维数组   归一化处理
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

# 由7转变为数组[0,0,0,0,0,0,0,1,0,0]
from tensorflow.keras.utils import to_categorical
print("before change:", test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after change:", test_labels[0])

# 训练  随机选取128个作为一组
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# verbose: 日志记录——0:静默不显示任何信息,1(default):输出进度条记录
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print(test_loss)
print('test_acc', test_acc)


#预测
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[1]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000, 28*28))
res = network.predict(test_images)

for i in range(res[1].shape[0]):
    if (res[1][i] == 1):
        print("the number for the picture is : ", i)
        break