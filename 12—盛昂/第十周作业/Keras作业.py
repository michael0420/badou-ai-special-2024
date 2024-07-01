#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
# 数据集的收集
(train_images,train_labels),(test_images,test_labels) =mnist.load_data()
# 数据集的归一化处理
train_images =train_images.reshape((60000,28*28))
train_images =train_images.astype('float32')/255
test_images =test_images.reshape((10000,28*28))
test_images =test_images.astype('float32')/255
# 神经网络的搭建
network =models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))#512为神经元节点个数
network.add(layers.Dense(10,activation="softmax"))
network.compile(optimizer ='rmsprop',loss="categorical_crossentropy",metrics=['accuracy'])#不懂

from tensorflow.keras.utils import to_categorical
train_labels =to_categorical(train_labels)
test_labels =to_categorical(test_labels)

# 神经网络的训练
network.fit(train_images,train_labels,epochs=5,batch_size=128)

# 测试数据的检测
test_loss,test_acc =network.evaluate(test_images,test_labels,verbose=1)
print(test_loss)
print('test_acc:', test_acc)

(train_images,train_labels),(test_images,test_labels)= mnist.load_data()
digit =test_images[1]
plt.imshow(digit,cmap=plt.cm.binary)
plt.show()
test_images =test_images.reshape((10000,28*28))
res =network.predict(test_images)
for i in range(res[1].shape[0]):
    if (res[1][i] ==1):
        print("the number for the picture is :",i)
        break

