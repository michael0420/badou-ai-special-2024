from  tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import layers

(train_images,train_labels) , (test_images,test_labels) = mnist.load_data()
print("训练集形状：",train_images.shape)
print("训练集标签形状：",train_labels.shape)
print("测试集形状" ,test_images.shape)
print("测试集标签形状" , test_labels.shape)

#------------------------------------------------

ditig = test_images[0]
plt.imshow(ditig,cmap='gray')

#----------------------------------

model = models.Sequential()
model.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
model.add(layers.Dense(10,activation='softmax'))
model.compile(optimizer='rmsprop',loss="categorical_crossentropy",metrics=['accuracy'])


#------------------------------------------------
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32')/255

#---------------------------------------------------
from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

print(train_images.shape)
print(train_labels.shape)
print(train_labels)
#--------------------------------------------------
model.fit(train_images,train_labels,epochs=5,batch_size=128)

#-----------------------------------------------
test_loss , test_acc = model.evaluate(test_images,test_labels,verbose=2)
print(test_loss)
print('test_acc', test_acc)
# #------------------------------------------------------
(train_images,train_labels) , (test_images,test_labels) = mnist.load_data()
number = test_images[9373]
plt.imshow(number,cmap='gray')
plt.show()

test_images = test_images[9373].reshape(1,28*28)
res = model.predict(test_images)
print(res.shape)
print(res)












