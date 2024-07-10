from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical


# 准备数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print("train_images.shape =",train_images.shape)
print("train_labels =",train_labels)
print("test_images.shape =",test_images.shape)
print("test_labels =",test_labels)


# digit = test_images[0]
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()

# 构造模型
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', loss = 'categorical_crossentropy', metrics=['accuracy'])

# 数据预处理
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

# 标签值和计算结果的映射关系
print("before chage:", test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after chage:", test_labels[0])

# 模型训练
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# 模型验证
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose =1)
print(test_loss)
print('test_acc', test_acc)

# 模型测试
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[0]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000, 28*28))
res = network.predict(test_images)
# print(res[1].shape)

for i in range(res[0].shape[0]):
    if (res[0][i] == 1):
        print("the number for the picture is :",i)
        break
