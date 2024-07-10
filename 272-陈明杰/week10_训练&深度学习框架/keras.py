from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.utils import to_categorical

#
# # 第一步：导入库中的手写数字数据集
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#
# # 第二步：构建神经网络
# # 初始化一个每层都是全连接的神经网络
# network = models.Sequential()
# # 512表示隐藏层有512个节点，activation：隐藏层的激活函数，input_shape：输入层的节点数，只需
# # 要声明输入的每一个样本有多少个维度，即每一个样本有多少个节点即可，即28*28个节点，input_shape=(28*28,)
# # 表示输入是一个元组，有28*28个元素
# network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
# # 添加一个输出层，有10个输出节点，激活函数为softmax，类似于归一化，可以看作是概率，即判断的
# # 结果是某一个数字的概率是多少
# network.add(layers.Dense(10, activation='softmax'))
# # 对神经网络进行编译：
# # 1、optimizer='rmsprop'：这指定了模型在训练过程中使用的优化器。rmsprop是一种常用的优化算法，
# # 特别适用于循环神经网络（RNNs）和其他深度学习模型。它通过调整每个参数的学习率来加速训练过程，
# # 并有助于在训练过程中避免陷入局部最小值。
# # 2、loss：损失函数，用来计算误差的，categorical_crossentropy：计算预测概率分布与真实概率分布之间的交叉熵。
# # 3、metrics='accuracy'：这指定了模型在训练和测试过程中要监控的性能指标。在这里，我们选择了accuracy作为
# # 监控指标，它表示模型正确分类样本的比例
# network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#
# # 归一化处理，并且要把图片的格式修改一下，因为训练函数的要求
# train_images = train_images.reshape((60000, 28 * 28))
# train_images = train_images.astype('float32') / 255
# test_images = test_images.reshape((10000, 28 * 28))
# test_images = test_images.astype('float32') / 255
#
# # 把图片对应的标记也做一个更改：
# # 目前所有图片的数字图案对应的是0到9。
# # 例如test_images[0]对应的是数字7的手写图案，那么其对应的标记test_labels[0]的值就是7。
# # 我们需要把数值7变成一个含有10个元素的数组，然后在第8个元素设置为1，其他元素设置为0。
# # 例如test_lables[0] 的值由7转变为数组[0,0,0,0,0,0,0,1,0,0] ---one hot
# from tensorflow.keras.utils import to_categorical
# # 因为18行的输出层使用softmax，26行compile的loss='categorical_crossentropy'，所以
# # 这里必须要对标签进行one hot处理，否则后面的fit训练函数会报错
# train_labels = to_categorical(train_labels)
# test_labels = to_categorical(test_labels)
#
# # 训练
# network.fit(train_images, train_labels, batch_size=128, epochs=5)
# # 测试
# loss, accuracy = network.evaluate(test_images, test_labels, batch_size=128, verbose=1)
# # 推理
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# test_images = test_images.reshape((10000, 28 * 28))
# # test_images=test_images.astype('float32')/255
# res = network.predict(test_images)
# print(res[1])


# 1、导入手写数字集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 2、格式化和归一化处理
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# 3、把labels转变为one hot格式
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 4、构建网络
network = models.Sequential()
# 添加输入层和隐藏层
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

# 5、编译神经网络
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# network.compile(optimizer=Adam(lr=1e-3),loss='categorical_crossentropy',metrics=['accuracy'])

# 6、训练
network.fit(train_images, train_labels, batch_size=128, epochs=5)
# network.fit_generator(generator,steps_per_epoch=max(1,len(lines)//batch_size),epochs=50,
#                       verbose=1,callbacks=[checkpoint,reduce_lr],validation_data=gen,
#                       validation_steps=max(1,len),initial_epoch=0)

# 7、测试
loss, accuracy = network.evaluate(test_images, test_labels, verbose=1)
print(f'loss={loss}')
print(f'accuracy={accuracy}')

# 8、推理
res = network.predict(test_images)
print(test_labels[1])
print(res[1])
