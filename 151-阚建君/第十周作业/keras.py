from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import cv2
'''
train_images是用于训练系统的手写数字图片;
train_labels是用于标注图片的信息;
test_images是用于检测系统训练效果的图片；
test_labels是test_images图片对应的数字标签。
'''
# 数据结果训练类别第一个是5   测试类别第一个数字是7
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images.shape = ',train_images.shape)
print('tran_labels = ', train_labels)
print('test_images.shape = ', test_images.shape)
print('test_labels', test_labels)

# 绘制第一个数字图像
digit = train_images[0]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

# 搭建keras网络
netWork = models.Sequential()
netWork.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
netWork.add(layers.Dense(10,activation='softmax'))
# 编译网络结构    loss使用交叉熵   评判标准使用正确率进行判定
netWork.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

# 训练前预处理
# 训练数据归一化

train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255
'''
使用onehot进行标记元素类别
'''
print("变换前数字类别：",test_labels[0])
print("变换前数字类别：",test_labels[1])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("变换后数字类别：",test_labels[0])
print("变换后数字类别：",test_labels[1])

# 训练网络数据
netWork.fit(train_images,train_labels,epochs=5,batch_size=128)

# 测试集测试网络数据
test_loss, test_acc = netWork.evaluate(test_images, test_labels, verbose=1)
print(test_loss)
print('test_acc', test_acc)

# 验证集  可以自己手写一个数字进行上传
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# digit = test_images[0]  # 展示当前第多少张图
# 读取和预处理自定义图像
pic_path = './my_own_4.png'
img = plt.imread(pic_path)

if pic_path[-4:] == '.png':  # .jpg图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算
    img = img * 255

# 转换为灰度图
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 调整大小到28x28
img = cv2.resize(img, (28, 28))

# 展示图像
plt.imshow(img, cmap=plt.cm.binary)
plt.show()

# 规范化和重塑图像
img = img.reshape((1, 28*28)).astype('float32') / 255

# 进行预测
res = netWork.predict(img)

# 输出预测结果
for i in range(res[0].shape[0]):  # 取范围0-10
    if (res[0][i] == 1):
        print("the number for the picture is : ", i)  # 预测当前图片的数字
        break










