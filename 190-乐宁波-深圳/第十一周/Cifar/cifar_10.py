import keras
from keras import layers, models
import cifar_10_data

max_steps = 4000
batch_size = 100
num_examples_for_eval = 10000
data_dir = 'cifar_data/cifar-10-batches-bin'

# 加载数据
train_dataset = cifar_10_data.inputs(data_dir=data_dir, batch_size=batch_size, distorted=True)
test_dataset = cifar_10_data.inputs(data_dir=data_dir, batch_size=batch_size, distorted=None)

# 构建CNN模型
model = models.Sequential()

# 卷积层一
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))

# 卷积层二
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# 卷积层三
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

# 展平层
model.add(layers.Flatten())

# 全连接层
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# 编译模型
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=5, validation_data=test_dataset)

# 评估
test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
print(f'Test accuracy: {test_acc}')

model.save('D:\\basic_cnn_model.h5')
model.summary()
