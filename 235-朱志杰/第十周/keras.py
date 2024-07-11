import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
# 记载训练集、测试集数据，labels是标签
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# digit = test_images[0]
# import matplotlib.pyplot as plt
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()

from tensorflow.keras import models
from tensorflow.keras import layers
# 创建模型
network = models.Sequential()
# 添加节点，这里是输入层，activation是激活函数类型
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
# 添加节点，这里是输出层，有10个节点，
network.add(layers.Dense(10, activation='softmax'))
# 编译模型
network.compile(optimizer='rmsprop', loss='categorical_crossentropy',
               metrics=['accuracy'])

# 数据预处理，转化为60000,28*28的一维数组
train_images = train_images.reshape((60000, 28*28))
# 归一化
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

# one-hot编码，向量的每个位置都对应一个类别，且只有一个位置的值是 1（表示该类别的存在），其余位置的值都是 0
from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 开始训练，epochs代表训练次数
network.fit(train_images, train_labels, epochs=5, batch_size = 128)

# 测试性能
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print(test_loss)
print('test_acc', test_acc)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[1]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000, 28*28))
# 预测图片
res = network.predict(test_images)

for i in range(res[1].shape[0]):
    if (res[1][i] == 1):
        print("the number for the picture is : ", i)
        break