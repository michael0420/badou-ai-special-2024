import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard

from tf.tf_graph import sess

# 加载 MNIST 数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 数据预处理
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 定义模型
model = Sequential()
model.add(Dense(200, activation='relu', input_shape=(28 * 28,)))
model.add(Dense(10, activation='softmax'))

# 编译模型
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 创建一个 TensorBoard 回调对象
tensorboard = TensorBoard(log_dir='logs', write_graph=True)

# 训练模型
model.fit(train_images, train_labels, epochs=5, batch_size=128, callbacks=[tensorboard])
file_writer = tf.summary.FileWriter('D:\python\pythonlearning\badoulearning\129-李彦松-北京\tf\logs', sess.graph)
# 测试模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)