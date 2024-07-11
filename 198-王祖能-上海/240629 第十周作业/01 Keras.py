from tensorflow.keras.datasets import mnist  # 手写数字样本库
import matplotlib.pyplot as plt
'''
【1】手写数字样本库的训练/检测数据加载到内存
'''
(train_sample, train_target), (test_sample, test_target) = mnist.load_data()  # 第一次运行需下载，慢
print(train_sample.shape, train_target.shape, test_sample.shape, test_target.shape)  # (60000，28，28) 6万张图片大小28*28
for i in range(3):
    plt.subplot(2, 3, i+1)
    # plt.imshow(train_sample[i], cmap='gray')  # 灰度图为'gray' ， 二值图为'binary'也可按此设置
    plt.imshow(train_sample[i], cmap=plt.cm.binary)  # 显示为黑白二值图片
    plt.title(train_target[i])  # 训练集前3张手写数字图片是5， 0， 4
    plt.subplot(2, 3, i+4)
    plt.imshow(test_sample[i], cmap='binary')
    plt.title(test_target[i])  # 测试集前3张手写数字图片是7， 2， 1
plt.xticks([]), plt.yticks([])
plt.show()
'''
【2】tensorflow.Keras搭建一个有效识别图案的神经网络
'''
from tensorflow.keras import models, layers
# 把所有数据处理层按序列串联起来，设置两个全连接层后 add 到当前串联的神经网络中
neural_network = models.Sequential()
layer1 = layers.Dense(100, activation='relu', input_shape=[28 * 28, ])  # Dense数据处理层为全连接层，易错：接收数据须是28*28的二维数组, 后面“,“表示数组里面的每一个元素到底包含多少个数字都没有关系
layer2 = layers.Dense(10, activation='softmax')  # 多分类任务采用softmax激活函数
neural_network.add(layer1)
neural_network.add(layer2)
# 网络编译，优化器：Root Mean Square Prop, 损失函数：(sparse_)categorical_crossentropy，度量方式：准确率
neural_network.compile(optimizer='rmsprop', loss=['categorical_crossentropy'], metrics=['accuracy'])
'''
【4】数据输入到网络模型之前，做归一化处理
'''
import numpy as np
train_sample = train_sample.reshape([60000, 28*28])  # 原来每个元素是28行，28列的二维数组，转变为一个28*28个元素的一维数组
train_sample = train_sample.astype(np.float16) / 255  # 数字图案是灰度图0到255之间，转变为0-1之间的浮点值
test_sample = test_sample.reshape([10000, 28*28])
test_sample = test_sample.astype('float16') / 255
'''
【5】输出数据做one hot处理成为矩阵形式
'''
from tensorflow.keras.utils import to_categorical
train_target = to_categorical(train_target)  # 把 target 标签的数值变成含10个元素的数组，对应数值位置的元素为1，其他为0。
print(train_target.shape, train_target[0])  # 例如test_lables[0] 的值由5转变为数组[0,0,0,0,0,1,0,0,0,0] ---one hot
'''
【6】训练模型，检查进度
'''
neural_network.fit(train_sample, train_target, batch_size=150, epochs=5)
loss, accuracy = neural_network.evaluate(train_sample, train_target, verbose=0)  # verbose=1指代进度条计算显示
print('loss:{}\naccuracy:{}\n'.format(loss, accuracy))
'''
【7】推理模型，检查结果
'''
# # test_sample = test_sample * 255  # 调用前面定义的数据要注意输出结果类型与输入float32一致
# # test_sample = test_sample.astype(np.uint8)
# (train_sample, train_target), (test_sample, test_target) = mnist.load_data()
# test_sample = test_sample.reshape([10000, 28*28])
res = neural_network.predict(test_sample, batch_size=300)
print(res[0])

for i in range(3):
    for index, value in enumerate(res[i]):  # for i in range(res[0].shape[0])中res[0].shape是二维数组
        if value == 1.0:
            print('推测第{}个数字为：{}'.format(i+1, index))
            break
