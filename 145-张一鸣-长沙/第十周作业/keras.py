# coding = utf-8

'''
        使用keras接口（底层tensorflow）实现训练：
        1.读取数据
        2.定义模型，填充输入层、隐藏层、输出层，构建模型，编译模型
        3.输入数据转换成一维，并归一化处理
        4.输出格式转换为 one hot
        5.执行训练 → fit()
        6.执行测试，查看结果 → evaluate()
        7.进行推理 → predict()
        8.输出 one hot 格式转换为可读结果
'''


from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


# 读取训练、测试数据（手写数字0-9）
(train_img, train_label), (test_img, test_label) = mnist.load_data()
print("训练集数据shape：", train_img.shape)
print("训练集标签：", train_label)
print("测试集数据shape：", test_img.shape)
print("测试集标签：", test_label)

# demo1 = train_img[0]
# demo2 = test_img[0]
# plt.imshow(demo1, cmap=plt.cm.binary)       # plt.cm.binary将图片转为二值图
# plt.imshow(demo2, cmap=plt.cm.binary)
# plt.axis('off')     # 关闭坐标轴显示
# plt.show()

# 创建训练模型
train_model = models.Sequential()       # Sequential()串联模式
train_model.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
# Dense-全连接模式，添加输入层（输入尺寸28*28）、隐藏层（512个神经元），激活函数为 relu
train_model.add(layers.Dense(10, activation='softmax'))
# 添加输出层（输出元素10，softmax 将输出转为 <1 的概率）
train_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# 编译模型，rmsprop梯度下降法，categorical_crossentropy交叉熵，accuracy主要监控的准确度

# 输入数据处理
train_img = train_img.reshape((60000, 28*28))   # reshape变成一维数组
train_img = train_img.astype('float32') / 255   # 像素值归一化

test_img = test_img.reshape((10000, 28*28))
test_img = test_img.astype('float32') / 255

# 输出结果以 one hot 表示
print('原输出：', train_label[0])
train_label = to_categorical(train_label)
test_label = to_categorical(test_label)
print('one hot 输出：', train_label[0])

# 执行训练
train_model.fit(train_img, train_label, epochs=8, batch_size=100)

# 用测试集查看训练结果
test_loss, test_accuracy = train_model.evaluate(test_img, test_label, verbose=1)        # verbose日志输出粒度
print('测试集损失：', test_loss)
print('测试集准确率：', test_accuracy)

# 进行推理（推理数据实际不与训练集、测试集相同，且数据与训练内容相关）
(train_img, train_label), (test_img, test_label) = mnist.load_data()
demo3 = test_img[80]
plt.imshow(demo3, cmap=plt.cm.binary)
plt.axis('off')
plt.show()

test_img = test_img.reshape((10000, 28*28))
result = train_model.predict(test_img)
print('推理结果shape：', result.shape)
print(result)

# 输出可读结果
for i in range(result[80].shape[0]):
    if result[80][i] == 1:
        print('图片推理为：', i)
        break

