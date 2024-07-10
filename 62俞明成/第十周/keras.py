from tensorflow.keras.datasets import mnist
from tensorflow.keras import models, layers
import matplotlib
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
import cv2 as cv

matplotlib.use('TkAgg')

(train_x, train_y), (test_x, test_y) = mnist.load_data()

one = train_x[0]
plt.imshow(one, cmap=plt.cm.binary)
plt.show()

network = models.Sequential(
    [
        layers.Dense(512, activation='relu', input_shape=(28 * 28,)),
        layers.Dense(10, activation='softmax')
    ]
)
network.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                metrics=['accuracy'])

train_x = train_x.reshape((60000, 28 * 28))
train_x = train_x.astype('float32') / 255

test_x = test_x.reshape((10000, 28 * 28))
test_x = test_x.astype('float32') / 255

# 转成 one-hot
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)

network.fit(train_x, train_y, epochs=5, batch_size=128)

test_loss, test_accuracy = network.evaluate(test_x, test_y, batch_size=128, verbose=1)
print(test_loss)
print('test_acc', test_accuracy)
print(network.summary())

x = cv.imread('7.jpg', 0)
resized_x = cv.resize(x, (28, 28), interpolation=cv.INTER_AREA)
resized_x = resized_x.astype('float32') / 255
plt.imshow(resized_x, cmap=plt.cm.binary)
plt.show()
reshape_x = resized_x.reshape((1, 28 * 28))

result = network.predict(reshape_x)
prediction = result.argmax()
print(f'result:{result}')
print(f'prediction:{prediction}')
