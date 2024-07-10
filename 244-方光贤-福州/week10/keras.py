from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

# 1.导入数据集 读取train和test标签对
(train_data, train_label), (test_data, test_label) = mnist.load_data()
print("train_data.shape = ", train_data.shape)
print("train_label = ", train_label)
print("test_data.shape = ", test_data.shape)
print("test_label = ", test_label)

digit = test_data[0]  # 打印测试
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

# 2. 使用sequential搭建神经网路
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
model.add(layers.Dense(10, activation='softmax'))

# 3. 配置模型
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. 数据归一化
train_data = train_data.reshape((60000, 28*28))
train_data = train_data.astype('float32') / 255
test_data = test_data.reshape((10000, 28*28))
test_data = test_data.astype('float32') / 255

# 5. one-hot转换
print("before change:", test_label[0])
train_label = to_categorical(train_label)
test_label = to_categorical(test_label)
print("after change: ", test_label[0])

# 6. 训练模型
model.fit(train_data, train_label, epochs=5, batch_size=128)
test_loss, test_acc = model.evaluate(test_data, test_label, verbose=1)
print('test_loss', test_loss)
print('test_acc', test_acc)

# 7. 预测
(train_data, train_label), (test_data, test_label) = mnist.load_data()
digit = test_data[1]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
test_data = test_data.reshape((10000, 28*28))
res = model.predict(test_data)

for i in range(res[1].shape[0]):
    if (res[1][i] == 1):
        print("the number for the picture is : ", i)
        break

