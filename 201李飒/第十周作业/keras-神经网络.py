from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# 整理训练数据,downlode训练数据和测试数据，数据归一化，转浮点数处理
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

x_train = train_images.reshape(60000,28*28)
x_train = x_train.astype('float32') / 255

x_test = test_images.reshape(10000,28*28)
x_test = x_test.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print(test_labels[0])

# 构建网络
network = models.Sequential()
network.add(layers.Dense(512, activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#进行网络训练
network.fit(x_train, train_labels, epochs=5, batch_size= 128)
# 进行验证
test_loss, test_acc = network.evaluate(x_test, test_labels, verbose=1)
print(test_loss)
print(test_acc)

# 带入一个手写图片进行识别，看它的识别效果
(train_iamges, train_labels), (test_images, test_labels)= mnist.load_data()
digit = test_images[9999]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

test_images = test_images.reshape(10000,28*28)
print(test_images.shape)
res = network.predict(test_images)
print(res)
for i in range(res[9999].shape[0]):
    if res[9999][i] == 1:
        print('The number is :',i)
        break




