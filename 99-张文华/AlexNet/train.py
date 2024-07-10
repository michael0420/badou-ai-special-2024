'''
进行模型训练，以及训练过程中的参数调整
'''
import numpy as np
from model import AlexNet
from tensorflow import keras
import cv2
import utils

# 将从文件中读取的数据，处理成训练所用的格式
def generate_arrays_from_file(lines, batch_size):
    # 获取长度
    n = len(lines)
    i = 0
    while True:
        x_train = []
        y_train = []
        # 获取一个batch_size大小的数据：
        for j in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)

            name = lines[i].split(';')[0]
            img = cv2.imread(f'data/image/train/{name}')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            x_train.append(img)
            y_train.append(lines[i].split(';')[1])
            # 读完一个周期后重新开始
            i = (i + 1) % n
        # 处理图像
        x_train = utils.resize_image(x_train, (224, 224))
        x_train = x_train.reshape(-1, 224, 224, 3)
        y_train = keras.utils.to_categorical(np.array(y_train), num_classes=2)
        yield x_train, y_train


if __name__ == '__main__':

    # 模型保存的路径，
    log_dir = './log/'

    # 打开数据集
    with open('./data/dataset.txt') as f:
        lines = f.readlines()

    # 打乱行，打乱的数据更利于训练
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 90%用于训练，10%用于推理
    num_val = int(len(lines) * 0.1)
    num_train = len(lines) - num_val

    # 实例化模型
    model = AlexNet.AlexNet()

    # 设定模型保存方式
    checkpoint_period1 = keras.callbacks.ModelCheckpoint(
        filepath=log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='acc',
        save_weights_only=False,
        save_best_only=True,
        period=3,
        verbose=0
    )

    # 学习率下降的方式，acc三次不下降就下降学习率继续学习
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='acc',
        factor=0.5,
        patience=3,
        verbose=0
    )

    # 设置早停
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=0
    )

    # 配置模型训练过程
    optimi = keras.optimizers.Adam(lr=1e-3)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimi, metrics=['accuracy']
    )

    # 训练集大小
    batch_size = 128

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(
        num_train, num_val, batch_size
    ))

    # 开始训练
    model.fit_generator(
        generate_arrays_from_file(lines[:num_train], batch_size),
        steps_per_epoch=max(1, num_train // batch_size),
        validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
        validation_steps=max(1, num_val//batch_size),
        epochs=50,
        initial_epoch=0,
        callbacks=[checkpoint_period1, reduce_lr]
    )
    model.save_weights(log_dir + 'train_model.h5')
