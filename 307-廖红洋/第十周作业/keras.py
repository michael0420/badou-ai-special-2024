# -*- coding: utf-8 -*-
"""
@File    :   keras.py
@Time    :   2024/06/29 16:19:38
@Author  :   廖红洋 
"""
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

# 数据读取，在处理之前取第4张进行展示
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
img = test_images[4]
plt.imshow(img, cmap=plt.cm.binary)

# 网络构建
network = models.Sequential()
network.add(
    layers.Dense(20, activation="relu", input_shape=(28 * 28,))
)  # 这里中间层设置20个节点，对于这种简单的图片分类，确实不需要512那么多的节点
network.add(layers.Dense(10, activation="softmax"))  # 多分类指定softmax，将输入映射到0-1的范围内
network.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)

# 数据集处理：划分，转换为tensor，并转为float32
train_images = train_images.reshape(60000, 28 * 28)
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape(10000, 28 * 28)
test_images = test_images.astype("float32") / 255

# 标签处理，转换为一个数组，每个下标存储一个类别
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 训练，实际上，训练以后需要保存模型，测试脚本与训练是分开的，只有需要的时候才会调参重新训练。
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# 推理
res = network.predict(test_images)  # 由于之前定义的keras张量格式限制，无法只推理一个图片，因此推理所有，并取其中第4张查看效果
for i in range(res[4].shape[0]):
    if res[4][i] > 0.5:
        print("the number for the picture is : ", i)
        break
plt.show()
