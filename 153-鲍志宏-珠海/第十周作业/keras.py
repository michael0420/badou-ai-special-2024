from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images.shape = ', train_images.shape)
print('tran_labels = ', train_labels)
print('test_images.shape = ', test_images.shape)
print('test_labels', test_labels)

#展示测试集中的一个图片
digit = test_images[0]
import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

#构建神经网络模型
from tensorflow.keras import models
from tensorflow.keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy',
               metrics=['accuracy'])

#数据预处理，将训练和测试图片展平成一维数组，并归一化到0-1范围。
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

from tensorflow.keras.utils import to_categorical
print("before change:" ,test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after change: ", test_labels[0])

#训练模型
network.fit(train_images, train_labels, epochs=5, batch_size = 128)

#模型评估
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print(test_loss)
print('test_acc', test_acc)

# 展示第二张测试图片并预测
digit = test_images[1].reshape(28, 28)  # 重新展示原始形状的图片
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

# 重新调整测试数据形状
test_images = test_images.reshape((10000, 28*28))

# 进行预测
res = network.predict(test_images)

# 找到预测结果中最高概率的索引
predicted_label = res[1].argmax()
print("The number for the picture is:", predicted_label)
