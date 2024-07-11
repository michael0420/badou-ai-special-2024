from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 手写数字识别
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 串联起每个数据处理层
network = models.Sequential()
# 构造一个数据处理层，输入层和隐藏层
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
# 构造一个输出层，softmax 表示输出层激活函数
network.add(layers.Dense(10, activation='softmax'))
# categorical_crossentropy 表示多分类交叉，metrics=['accuracy']指定评估指标为准确率
network.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
# 将标量转换为二进制向量
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 将数据输入网络进行训练
network.fit(train_images, train_labels, epochs=5, batch_size=128)
# test_loss：损失值；test_acc:准确率
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)

# 测试识别效果
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[1]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000, 28 * 28))
res = network.predict(test_images)

for i in range(res[1].shape[0]):
    if (res[1][i] == 1):
        print("the number for the picture is : ", i)
        break
