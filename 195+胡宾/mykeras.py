
import numpy as np

import cv2
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

# 加载训练数据，测试数据
(training_data, training_label), (verification_data, verification_label) = mnist.load_data()

# 构建神经网络
sequential = models.Sequential()
sequential.add(layers.Dense(618, activation='sigmoid', input_shape=(28 * 28,)))
sequential.add(layers.Dense(618, activation='sigmoid'))
sequential.add(layers.Dense(10, activation='softmax'))

# 编译
sequential.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 处理输入数据，做归一化处理
training_data = training_data.reshape((60000, 28 * 28))
training_data = training_data.astype('float32') / 255

verification_data = verification_data.reshape((10000, 28 * 28))
verification_data = verification_data.astype('float32') / 255
# 标签处理
training_label = to_categorical(training_label)
verification_label = to_categorical(verification_label)
# 进行训练
sequential.fit(training_data, training_label, epochs=6, batch_size=618)
'''
测试数据输入，检验网络学习后的图片识别效果.
识别效果与硬件有关（CPU/GPU）.
'''
sequential.evaluate(verification_data, verification_label, verbose=1)

# 载入模型
test_tuili = cv2.imread("tuilituxianchagn.png", 0)
resize = cv2.resize(test_tuili, (28, 28), interpolation=cv2.INTER_CUBIC)
test_tuili = resize.reshape((1, 28 * 28))
res = sequential.predict(test_tuili)
tolist = res[0].flatten().tolist()
i = tolist.index(np.max(tolist))
# max_val = np.max(res[0].shape[0])
# i = [index for index, item in enumerate(res[0]) if item == max_val]
print("the number for the picture is : ", i)
# print(res.(1))
# for i in range(res[0].shape[0]):
#     if (res[0][i] == 1):
#         print("the number for the picture is : ", i)
#         break
