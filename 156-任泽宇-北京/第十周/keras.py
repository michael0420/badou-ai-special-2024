from tensorflow.keras.datasets import mnist
from tensorflow.keras import models, layers, utils
import matplotlib.pylab as plt
import cv2, os

# 加载数据
(train_img, train_label), (test_img, test_label) = mnist.load_data()

# 查看测试集中数据
img1 = test_img[0]
plt.imshow(img1, cmap=plt.cm.binary)
plt.show()

# 建立一个空神经网络模型
network = models.Sequential()
# 增加隐藏层和输出层
network.add(layers.Dense(512, activation='relu', input_shape=(28*28, )))
network.add(layers.Dense(10, activation='softmax'))

# 将构建好的网络进行编译
# categorical_crossentropy 交叉熵
# metrics 判断正确式的方式      accuracy：正确率
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 进行归一化
train_img = train_img.reshape((60000, 28*28))
train_img = train_img.astype('float32') / 255

test_img = test_img.reshape((10000, 28*28))
test_img = test_img.astype('float32') / 255

# 进行one hot编码
train_label = utils.to_categorical(train_label)
test_label = utils.to_categorical(test_label)

# 训练
network.fit(train_img, train_label, epochs=5, batch_size=128)

# 测试数据输入后识别效果
test_loss, test_acc = network.evaluate(test_img, test_label, verbose=1)

# 输入一张图片，查看识别效果
img_list = os.listdir('../../imgs/digit_shouxie')
# zhenshi = [6, 2, 3, 3, 6, 4, 5]
for i in range(len(img_list)):
    digit = cv2.imread(f'../../imgs/digit_shouxie/{img_list[i]}', 0)
    # digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_CUBIC)
    plt.imshow(digit, cmap=plt.cm.binary)
    plt.show()
    digit = digit.reshape((1, 28*28))
    res = network.predict(digit)
    print(res)
    print(res.shape)
    print(res.shape[1])
    for j in range(res.shape[1]):
        if (res[0][j] == 1):
            print("the number for the picture is : ", j)
            break




