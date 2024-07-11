# Author: Zhenfei Lu
# Created Date: 6/25/2024
# Version: 1.0
# Email contact: luzhenfei_2017@163.com, zhenfeil@usc.edu

from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import layers
import random
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


class TensorflowClassificationModel(object):
    def __init__(self, inputDim=28*28):
        self.network = models.Sequential()
        self.network.add(layers.Dense(512, activation='relu', input_shape=(inputDim,)))
        self.network.add(layers.Dense(10, activation='softmax'))

    def forward(self, x):
        x = self.network.predict(x)
        return x

    def calLoss(self, y_predict, y):
        return self.loss(y_predict, y)

    def data_iter(self, batch_size, features, labels):
        num_examples = len(features)
        indices = list(range(num_examples))
        random.shuffle(indices)
        for i in range(0, num_examples, batch_size):
            batch_indices = tf.tensor(indices[i: min(i + batch_size, num_examples)])
            yield features[batch_indices], labels[batch_indices]

    def fit(self, x, y, epoch=5, learning_rate=0.001, batch_size=128):
        self.network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        hist = self.network.fit(x, y, epochs=epoch, batch_size=batch_size)
        test_loss, test_acc = test_loss, test_acc = self.network.evaluate(test_images, test_labels, verbose=1)



if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    model = TensorflowClassificationModel(train_images.shape[1] * train_images.shape[2])
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255

    model.fit(train_images, train_labels)
