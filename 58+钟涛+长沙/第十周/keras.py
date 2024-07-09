from PIL import Image
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

#加载数据
(train_datas,train_lables),(test_datas,test_lables)=mnist.load_data()

#打印图片
digit = test_datas[0]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()


#构建神经网络
from tensorflow.keras import models
from tensorflow.keras import layers

network = models.Sequential()
#隐藏层
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
#输出层
network.add(layers.Dense(10, activation='softmax'))

#编译神经网络  categorical_crossentropy
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])


#归一化
train_datas = train_datas.reshape(60000, 28*28)
train_datas = train_datas.astype('float32') / 255 * 0.99 + 0.01
test_datas = test_datas.reshape(10000,28 * 28)
test_datas = test_datas.astype('float32') / 255 * 0.99 + 0.01

#标签处理，使其变为[0,0,0,0,0,1,0,0] 格式
from tensorflow.keras.utils import to_categorical

print("before change:" ,test_lables[0])
train_lables = to_categorical(train_lables)
test_lables = to_categorical(test_lables)
print("after change: ", test_lables[0])
#x训练
network.fit(train_datas,train_lables, epochs = 5, batch_size = 128)

#测试
"""
network.evaluate(test_datas, test_labels) 是用于评估已经训练好的神经网络模型在测试数据上的性能的代码。这个方法会返回模型在测试数据上的损失值和指定的评估指标值（例如准确率）。
参数解释
test_datas: 这是测试数据集，通常是一个包含输入特征的数组或矩阵。
test_labels: 这是测试数据集的标签，通常是一个包含真实标签的数组或矩阵。
verbose=1: 这是一个控制评估过程中输出信息的参数。当 verbose=1 时，评估过程中会显示进度条和评估结果。
返回值
evaluate 方法返回一个包含两个值的列表：

损失值：这是模型在测试数据上的损失函数值，例如 categorical_crossentropy。
评估指标值：这是模型在测试数据上的评估指标值，例如 accuracy。
"""
test_loss, test_acc = network.evaluate(test_datas,test_lables,verbose=1)

#推理,预测
(train_images,train_lables),(test_images,test_lables)=mnist.load_data()
digit = test_images[1]
import cv2
# img = Image.open("dataset/my_own_image.png").convert('L')
# digit = np.array(img)

# digit = cv2.imread('dataset/my_own_6.png')
# digit = cv2.cvtColor(digit, cv2.COLOR_RGB2GRAY)

plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
print("digit shape :", digit.shape)
digit = digit.reshape((1,28*28))
print("digit: ", digit)
res = network.predict(digit)
print("res: ", res)
label = np.argmax(res)
print("网络认为图片的数字是：", label)

for i in range(res[0].shape[0]):
    if res[0][i] == 1:
        print("图片的数字是：", i)
        break