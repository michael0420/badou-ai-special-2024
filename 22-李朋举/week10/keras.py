# [1] 加载数据
# tensorflow.keras 存在版本依赖关系(如1.14-2.14)   keras-可单独指定版本  mnist->手写数字的数据
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
'''
将训练数据和检测数据加载到内存中(第一次运行需要下载数据，会比较慢):
1.train_images是用于训练系统的手写数字图片:
     train_images.shape -> (60000, 28, 28)，train_images是一个含有60000个元素(图片)的数组.数组中的元素是一个二维数组，二维数组的行和列都是28.
                            也就是说，一个数字图片的大小是28*28.
train_labels是用于标注图片的信息;
     train_lables -> [5 0 4 ... 5 6 8] 第一张手写数字图片的内容是数字5，第二种图片是数字0，以此类推.
test_images是用于检测系统训练效果的图片；
     test_images.shape ->  (10000, 28, 28)  用于检验训练效果的图片有10000张
test_labels是test_images图片对应的数字标签。
     test_labels -> [7 2 1 ... 4 5 6] 用于检测的第一张图片内容是数字7，第二张是数字2，依次类推
'''

# [2]创建模型
from tensorflow.keras import models
from tensorflow.keras import layers

'''
使用tensorflow.Keras搭建一个有效识别图案的神经网络，
   models.Sequential():表示把每一个数据处理层串联起来. 按顺序添加神经网络的各个层，而无需显式地定义模型的结构。可以方便地将多个层堆叠在一起，形成一个深度学习模型。
'''
network = models.Sequential()
'''
Dense - 密集的
使用 TensorFlow 的 API（Keras）来定义一个神经网络模型时，向模型中添加一个密集连接层（全连接层）的操作。
    `network.add(layers.Dense(512, activation='relu'， input_shape=(28*28,)))`: 
          `network` 是之前创建的 Sequential 模型对象，`add()` 方法用于向模型中添加新的层。
          `layers.Dense(512, activation='relu')` layers:表示神经网络中的一个数据处理层。(dense:全连接层)，添加一个具有 512 个神经元的密集连接层，激活函数使用`ReLU`。
          `input_shape=(28*28,)`: 这部分是指定该密集连接层的输入形状，输入是一个二维张量，形状为`(28*28,)`，即 28 行乘以 28 列的一个矩阵。
                                  因为在 MNIST 数字识别任务中，输入图像的尺寸通常是 28x28 的像素。后面的“,“表示数组里面的每一个元素到底包含多少个数字都没有关系.
总体来说，这行代码的作用是在模型中添加一个具有 512 个神经元的密集连接层(隐藏层)，该层的输入形状为 28x28 的二维张量，这个层将对输入数据进行特征提取和非线性变换。
                输入         隐藏层
               28X28         512   
'''
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
'''
在模型中添加一个具有 10 个神经元的密集连接层    ( 多分类-数字0-9 对应 10个类别(10个节点)   二分类-0或1  对应 2个类别(2个节点)  )
                输入         隐藏层      输出层
               28X28         512        10
'''
network.add(layers.Dense(10, activation='softmax'))
'''
使用 TensorFlow 的 API（Keras）来编译神经网络模型。
`network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])`:
         `compile()` 方法用于编译模型，以便可以在训练和预测时使用它。
         `optimizer='rmsprop'` 表示优化器使用随机梯度下降（Stochastic Gradient Descent，SGD）的一种变体——均方根传播（Root Mean Square Propagation， RMSProp）。
         `loss='categorical_crossentropy'` 表示损失函数使用交叉熵函数，这是一种常用的分类任务损失函数，它可以衡量预测值与真实值之间的差异。
         `metrics=['accuracy']`: `metrics` 参数指定了在训练过程中要监控的评估指标, 表示预测结果的准确率。
                                  误差:预测结果与正确结果的差距   准确率:在误差范围内有多少是预测对的                         
总体来说，这行代码的作用是编译模型，选择优化器、损失函数和评估指标，以便在训练和预测时使用。
'''
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# [3] 数据处理
'''
在把数据输入到网络模型之前，把数据做归一化处理:
1.reshape(60000, 28*28）:train_images数组原来含有60000个元素，每个元素是一个28行，28列的二维数组 (60000, 28, 28)
                         现在把每个二维数组转变为一个含有28*28个元素的一维数组 (60000, 28*28)
    在处理图像数据时，通常需要将二维图像数据转换为一维向量，以便后续的处理和训练。这个过程被称为"展平"（Flatten）或"一维化"（One-hot encoding）。
2.由于数字图案是一个灰度图，图片中每个像素点值的大小范围在0到255之间.
                        train_images.astype(“float32”)/255 把每个像素点的值从范围0-255转变为范围在0-1之间的浮点值。                         
这种展平操作通常在图像分类任务中使用，因为神经网络等模型通常需要输入是一维向量。通过展平图像数据，可以将二维图像转换为适合模型输入的一维向量形式。
'''
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

'''
to_categorical：用于将整数型标签转换为分类格式。
               它将整数标签值映射到一个固定大小的分类空间中，分类格式是一种表示类别信息的方式，它将每个类别表示为一个独热向量（One-hot vector）。
                输出层10个节点       10分类
        `train_labels` 是一个整数数组，它表示了训练数据的标签,目前所有图片的数字图案对应的是0到9
                           标签 0、1、2、3、4、5、6、7、8、9 等整数，
                           类别 0、1、2、3、4、5、6、7、8、9  
                                                  7->[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
                                                    如果 train_labels[0] = 7 (即train_labels中第一张图片的标签是7)       标签               
                                                    =>  train_labels[0] = 7  =》 [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]    类别 ---one hot
'''
from tensorflow.keras.utils import to_categorical

print("before change:", test_labels[0])  # 7
train_labels = to_categorical(train_labels)  # (60000, 10)
test_labels = to_categorical(test_labels)  # (10000, 10)
print("after change: ", test_labels[0])  # [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]

# [4] 训练模型
'''
把数据输入网络进行训练：
train_images：用于训练的手写数字图片；
train_labels：对应的是图片的标记；
batch_size：每次网络从输入的图片数组中随机选取128个作为一组进行计算。
epochs:每次计算的循环是五次
'''
network.fit(train_images, train_labels, epochs=5, batch_size=128)
'''                  
batchSize = 128，60000图片就有60000/128=468.75 即469个batchs可用
Epoch 1/5
469/469 [==============================] - 3s 6ms/step - loss: 0.2543 - accuracy: 0.9256
Epoch 2/5
469/469 [==============================] - 3s 6ms/step - loss: 0.1035 - accuracy: 0.9689
Epoch 3/5
469/469 [==============================] - 3s 6ms/step - loss: 0.0680 - accuracy: 0.9797
Epoch 4/5
469/469 [==============================] - 3s 6ms/step - loss: 0.0493 - accuracy: 0.9852
Epoch 5/5
469/469 [==============================] - 3s 7ms/step - loss: 0.0370 - accuracy: 0.9888
'''

# [5] 测试模型
'''
测试数据输入，检验网络学习后的图片识别效果.识别效果与硬件有关（CPU/GPU）.
verbose = 1 是否打印
'''
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print(test_loss)  # 0.07041556388139725
print('test_acc', test_acc)  # 0.9797999858856201
'''
测试结果：判断业务是否能上线。 如果存在过拟合的情况，loss相对于训练较高，accuracy相对于训练较低。
313/313 [==============================] - 1s 2ms/step - loss: 0.0704 - accuracy: 0.9798
'''

# [6] 推理
'''
输入一张手写数字图片到网络中，看看它的识别效果
'''
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[1]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000, 28 * 28))
res = network.predict(test_images)
'''
res (10000,10)  
[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.] ,[0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]...
'''

for i in range(res[1].shape[0]):
    if res[1][i] == 1:
        print("the number for the picture is : ", i)
        break