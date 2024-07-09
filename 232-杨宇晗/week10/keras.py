import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# 加载数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 显示第一个测试图像
plt.imshow(test_images[0], cmap=plt.cm.binary)
plt.show()

# 预处理数据
train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255

# 标签进行独热编码
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 定义模型
network = models.Sequential([
    layers.Dense(512, activation='relu', input_shape=(28 * 28,)),
    layers.Dense(10, activation='softmax')
])

# 编译模型
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# 训练模型
network.fit(train_images, train_labels, epochs=10, batch_size=128)

# 评估模型
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# 显示第二个测试图像
plt.imshow(test_images[1].reshape(28, 28), cmap=plt.cm.binary)
plt.show()

# 预测
predictions = network.predict(test_images)

# 输出预测结果
print("The number for the picture is:", predictions[1].argmax())
