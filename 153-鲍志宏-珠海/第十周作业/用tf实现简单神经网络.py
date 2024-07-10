import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# 加载数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 展示数据集中的一张图片
digit = test_images[0]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

# 数据预处理
train_images = train_images.reshape((60000, 28 * 28))
test_images = test_images.reshape((10000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# 将标签进行one-hot编码
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# 构建神经网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(28 * 28,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(train_images, train_labels, epochs=5, batch_size=128)

# 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

# 预测并展示第二张测试图片
digit = test_images[1].reshape(28, 28)
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

# 进行预测
predictions = model.predict(test_images)

# 找到预测结果中最高概率的索引
predicted_label = predictions[1].argmax()
print("The number for the picture is:", predicted_label)
