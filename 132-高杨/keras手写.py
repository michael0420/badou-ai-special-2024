'''
经典数据集，28*28

'''
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist

(train_img,train_lables),(test_img,test_labels) = mnist.load_data()

from tensorflow.keras import layers,models
network = models.Sequential()
network.add(layers.Dense(200,'relu',input_shape=(28*28,)))
network.add(layers.Dense(10,'softmax'))
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

train_img = train_img.reshape((60000,28*28))
train_img = train_img.astype('float')/255

test_img = test_img.reshape((10000,28*28))
test_img = test_img.astype('float')/255

from tensorflow.keras.utils import to_categorical
train_lables = to_categorical(train_lables)
test_labels = to_categorical(test_labels)
#进行训练
network.fit(train_img,train_lables,128,7)
test_loss,test_acc = network.evaluate(test_img,test_labels,verbose=1)
print(test_loss)
print("准确率是： ",test_acc)


(trian_img,train_labels),(test_img,test_lables) = mnist.load_data()

test_img = test_img.reshape((10000,28*28))
res = network.predict(test_img)
for i in range(res[1].shape[0]):
    if (res[1][i]==1):
        print('right, the number is :',i)
        break



(trian_img,train_labels),(test_img,test_lables) = mnist.load_data()
show_img = test_img[0]
plt.imshow(show_img,cmap=plt.cm.binary)
plt.show()

'''
第三步：
    用tensorflow.keras搭建一个有效识别图案的神经网络
'''
from tensorflow.keras import layers,models
content = models.Sequential()
content.add(layers.Dense(512,'relu',input_shape=(28*28,)))
content.add(layers.Dense(10,'softmax'))
content.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

trian_img = trian_img.reshape((60000,28*28))
trian_img = trian_img.astype('float') / 255

test_img = test_img.reshape((10000,28*28))
test_img = test_img.astype('float') / 255

from tensorflow.keras.utils import to_categorical
print("before_change: ",test_lables[0])
train_labels = to_categorical(train_labels)
test_lables = to_categorical(test_lables)
print("after change: ",test_lables[0])

#把数据输入网络进行训练，
content.fit(trian_img,train_labels,128,epochs=7)
#verbose 是否要打印的意思
test_loss,test_acc = content.evaluate(test_img,test_lables,verbose=1)
print(test_loss)
print('test_acc: ',test_acc)


(trian_img,train_labels),(test_img,test_lables) = mnist.load_data()

test_img = test_img.reshape((10000,28*28))
res = content.predict(test_img)
for i in range(res[1].shape[0]):
    if (res[1][i]==1):
        print('right, the number is :',i)
        break








