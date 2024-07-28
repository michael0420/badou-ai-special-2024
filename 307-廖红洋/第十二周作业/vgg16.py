# -*- coding: utf-8 -*-
"""
@File    :   vgg16.py
@Time    :   2024/07/13 20:57:43
@Author  :   廖红洋 
"""
import numpy as np
import time
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def normalization(x_img_train, x_img_test):
    mean = np.mean(x_img_train, axis=(0, 1, 2, 3))  # 四个维度 批数 像素x像素 通道数
    std = np.std(x_img_train, axis=(0, 1, 2, 3))
    # 测试集做一致的标准化 用到的均值和标准差 服从train的分布（有信息杂糅的可能）
    x_img_train = (x_img_train - mean) / (std + 1e-7)  # trick 加小数点 避免出现整数
    x_img_test = (x_img_test - mean) / (std + 1e-7)

    return x_img_train, x_img_test


# 数据读取
def load_images():
    (x_img_train, y_label_train), (x_img_test, y_label_test) = cifar10.load_data()

    x_img_train = x_img_train.astype(np.float32)  # 数据类型转换
    x_img_test = x_img_test.astype(np.float32)

    (x_img_train, x_img_test) = normalization(x_img_train, x_img_test)

    y_label_train = to_categorical(y_label_train, 10)  # one-hot
    y_label_test = to_categorical(y_label_test, 10)

    return x_img_train, y_label_train, x_img_test, y_label_test


class ConvBNRelu(tf.keras.Model):
    def __init__(
        self,
        filters,
        kernel_size=3,
        strides=1,
        padding="SAME",
        weight_decay=0.0005,
        rate=0.4,
        drop=True,
    ):
        super(ConvBNRelu, self).__init__()
        self.drop = drop
        self.conv = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        )
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.dropOut = keras.layers.Dropout(rate=rate)

    def call(self, inputs):  # , training=False
        layer = self.conv(inputs)
        layer = tf.nn.relu(layer)
        layer = self.batchnorm(layer)

        # 用来控制conv是否有dropout层，对应类ConvBNRelu中的self.drop属性
        if self.drop:
            layer = self.dropOut(layer)

        return layer


class VGG16Model(tf.keras.Model):
    def __init__(self):
        super(VGG16Model, self).__init__()
        self.conv1 = ConvBNRelu(filters=64, kernel_size=[3, 3], rate=0.3)
        self.conv2 = ConvBNRelu(filters=64, kernel_size=[3, 3], drop=False)
        self.maxPooling1 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv3 = ConvBNRelu(filters=128, kernel_size=[3, 3])
        self.conv4 = ConvBNRelu(filters=128, kernel_size=[3, 3], drop=False)
        self.maxPooling2 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv5 = ConvBNRelu(filters=256, kernel_size=[3, 3])
        self.conv6 = ConvBNRelu(filters=256, kernel_size=[3, 3])
        self.conv7 = ConvBNRelu(filters=256, kernel_size=[3, 3], drop=False)
        self.maxPooling3 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv11 = ConvBNRelu(filters=512, kernel_size=[3, 3])
        self.conv12 = ConvBNRelu(filters=512, kernel_size=[3, 3])
        self.conv13 = ConvBNRelu(filters=512, kernel_size=[3, 3], drop=False)
        self.maxPooling5 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv14 = ConvBNRelu(filters=512, kernel_size=[3, 3])
        self.conv15 = ConvBNRelu(filters=512, kernel_size=[3, 3])
        self.conv16 = ConvBNRelu(filters=512, kernel_size=[3, 3], drop=False)
        self.maxPooling6 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flat = keras.layers.Flatten()
        self.dropOut = keras.layers.Dropout(rate=0.5)

        self.dense1 = keras.layers.Dense(
            units=512,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(0.0005),
        )
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.dense2 = keras.layers.Dense(units=10)
        self.softmax = keras.layers.Activation("softmax")

    def call(self, inputs):  # , training=False
        net = self.conv1(inputs)
        net = self.conv2(net)
        net = self.maxPooling1(net)
        net = self.conv3(net)
        net = self.conv4(net)
        net = self.maxPooling2(net)
        net = self.conv5(net)
        net = self.conv6(net)
        net = self.conv7(net)
        net = self.maxPooling3(net)
        net = self.conv11(net)
        net = self.conv12(net)
        net = self.conv13(net)
        net = self.maxPooling5(net)
        net = self.conv14(net)
        net = self.conv15(net)
        net = self.conv16(net)
        net = self.maxPooling6(net)
        net = self.dropOut(net)
        net = self.flat(net)
        net = self.dense1(net)
        net = self.batchnorm(net)
        net = self.dropOut(net)
        net = self.dense2(net)
        net = self.softmax(net)
        return net


# 准备训练
if __name__ == "__main__":
    print("tf.__version__:", tf.__version__)
    print("keras.__version__:", keras.__version__)

    # 超参数
    training_epochs = 100
    batch_size = 256
    learning_rate = 0.1
    momentum = 0.9  # SGD加速动量
    weight_decay = 1e-6  # 权重衰减
    lr_drop = 20  # 衰减倍数

    tf.random.set_seed(2022)  # 固定随机种子，可复现

    def lr_scheduler(epoch):  # 动态学习率衰减，epoch越大，lr衰减越剧烈。
        return learning_rate * (0.5 ** (epoch // lr_drop))

    reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

    x_img_train, y_label_train, x_img_test, y_label_test = load_images()

    datagen = ImageDataGenerator(
        featurewise_center=False,  # 布尔值。将输入数据的均值设置为 0，逐特征进行。
        samplewise_center=False,  # 布尔值。将每个样本的均值设置为 0。
        featurewise_std_normalization=False,  # 布尔值。将输入除以数据标准差，逐特征进行。
        samplewise_std_normalization=False,  # 布尔值。将每个输入除以其标准差。
        zca_whitening=False,  # 布尔值。是否应用 ZCA 白化。
        rotation_range=15,  # 整数。随机旋转的度数范围 (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # 布尔值。随机水平翻转。
        vertical_flip=False,
    )  # 布尔值。随机垂直翻转。

    datagen.fit(x_img_train)

    model = VGG16Model()  # 调用模型

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate,
        decay=weight_decay,
        momentum=momentum,
        nesterov=True,
    )
    # 交叉熵、优化器，评价标准。
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    t1 = time.time()
    history = model.fit(
        datagen.flow(x_img_train, y_label_train, batch_size=batch_size),
        epochs=training_epochs,
        verbose=2,
        callbacks=[reduce_lr],
        steps_per_epoch=x_img_train.shape[0] // batch_size,
        validation_data=(x_img_test, y_label_test),
    )
    t2 = time.time()
    CNNfit = float(t2 - t1)
    print("Time taken: {} seconds".format(CNNfit))

    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.subplot(1, 2, 1)
    plt.plot(accuracy, label="Training Accuracy")
    plt.plot(val_accuracy, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.savefig("./results.png")
    plt.show()
