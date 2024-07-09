import numpy

# 读取数据
with open('dataset/mnist_train.csv', 'r') as train_data_file:
    train_data_list = train_data_file.readlines()
trainImages = []
trainLables = []
for record in train_data_list:
    all_values = record.split(',')
    trainImages.append(all_values[1:])
    trainLables.append(all_values[0])
with open('dataset/mnist_test.csv', 'r') as test_data_file:
    test_data_list = test_data_file.readlines()
testImages = []
testLables = []
for record in test_data_list:
    all_values = record.split(',')
    testImages.append(all_values[1:])
    testLables.append(all_values[0])

# 构建网络模型
from tensorflow.keras import models
from tensorflow.keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 数据归一化
trainImages = numpy.asfarray(trainImages) / 255.0 * 0.999 + 0.001
# trainImages=trainImages.reshape(len(trainImages),)
testImages = numpy.asfarray(testImages) / 255.0 * 0.999 + 0.001
# 标记进行onehot编码
from tensorflow.keras.utils import to_categorical

trainLables = to_categorical(trainLables)
testLables = to_categorical(testLables)
# 网络训练
network.fit(trainImages, trainLables, epochs=5, batch_size=128)
# 网络验证
testLoss, testAcc = network.evaluate(testImages, testLables, verbose=1)
# 网络测试
res = network.predict(testImages)
print('预测值：', numpy.argmax(res[0]))
print('真实值：', numpy.argmax(testLables[0]))
