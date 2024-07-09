import os

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


def generator_from_file(file_names, batch):
    file_num = len(file_names)
    i = 0
    while 1:
        images = []
        labels = []
        for j in range(batch):
            if i == 0:
                np.random.shuffle(file_names)
            filename = file_names[i]
            if filename.startswith("cat"):
                labels.append(0)
            elif filename.startswith("dog"):
                labels.append(1)
            else:
                continue
            img = cv2.imread(os.path.join(path, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255
            images.append(img)
            i = (i + 1) % file_num
        image_batch = resize_images(images, (224, 224))
        image_batch = image_batch.reshape(-1, 224, 224, 3)
        label_batch = keras.utils.to_categorical(labels, num_classes=2)
        yield (image_batch, label_batch)


def resize_images(images, size):
    with tf.name_scope('resize_image'):
        tmp = []
        for img in images:
            tmp.append(cv2.resize(img, size))
        return np.array(tmp)


def create_alexnet(input_shape, output_shape):
    model = keras.models.Sequential()
    # 第一个卷积层
    model.add(keras.layers.Conv2D(filters=48, kernel_size=(11, 11), strides=(4, 4), input_shape=input_shape,
                                  activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
    # 第二个卷积层
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
    # 第三、四个卷积层
    model.add(keras.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    # 第五个卷积层
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(output_shape, activation='softmax'))
    return model


if __name__ == '__main__':
    # 将数据放到当前目录的cat-dog-dataset文件夹下
    path = "cat-dog-dataset"
    files = os.listdir(path)
    batch_size = 128
    # images, labels = generator_from_file(files, batch_size=batch_size)
    my_model = create_alexnet((224, 224, 3), 2)

    # 定义各种回调函数
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='acc', factor=.5, patience=5, verbose=1)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    check_point = keras.callbacks.ModelCheckpoint('./ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                                  monitor='acc', save_weights_only=False, save_best_only=True, period=3)

    # 编译模型
    my_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                     metrics=['accuracy'])

    num_val = int(len(files) * 0.1)
    num_train = len(files) - num_val
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    my_model.fit_generator(generator_from_file(files[:num_train], batch_size),
                           steps_per_epoch=max(1, num_train // batch_size),
                           validation_data=generator_from_file(files[:-num_val], batch_size),
                           validation_steps=max((1, num_val // batch_size)),
                           epochs=50, initial_epoch=0, callbacks=[reduce_lr, early_stop])
    my_model.save_weights('./last.h5')
