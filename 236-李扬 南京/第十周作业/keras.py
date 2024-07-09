from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

#加载数据集
(train_images, trainlables), (test_images, test_lables) = mnist.load_data()
print(train_images.shape)
print(trainlables)
print(test_images.shape)
print(test_lables)

#打印第一张数据
digit = test_images[0]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

#构建神经网络
#构建输出层、隐藏层、输出层
networks = models.Sequential()
networks.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
networks.add(layers.Dense(10, activation='softmax'))
#编译
networks.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#数据归一化,调整成0-1之间
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float') / 255

#转换成one-hot独热编码
trainlables = to_categorical(trainlables)
test_lables = to_categorical(test_lables)

#进行网络训练
networks.fit(train_images, trainlables, epochs=5,batch_size=128)

#使用训练数据评估模型
test_loss, test_acc = networks.evaluate(test_images, test_lables, verbose=1)
print(test_loss)
print(test_acc)

#随机测试一张图片,验证预测结果
(train_images, trainlables), (test_images, test_lables) = mnist.load_data()
digit = test_images[2]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000, 28*28))
res = networks.predict(test_images)

for i in range(res[2].shape[0]):
    if(res[2][i] == 1):
        print("the number is : ", i)
        break

