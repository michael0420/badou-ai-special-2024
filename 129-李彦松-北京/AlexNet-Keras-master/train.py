from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils
from tensorflow.keras.optimizers import Adam
from model.AlexNet import AlexNet
import numpy as np
import utils
import cv2
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')


def generate_arrays_from_file(lines,batch_size):
    # 获取总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # 从文件中读取图像
            img = cv2.imread(r"./data/image/train" + '/' + name)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = img/255
            X_train.append(img)
            Y_train.append(lines[i].split(';')[1])
            # 读完一个周期后重新开始
            i = (i+1) % n
        # 处理图像
        X_train = utils.resize_image(X_train,(224,224))
        X_train = X_train.reshape(-1,224,224,3)
        Y_train = np_utils.to_categorical(np.array(Y_train),num_classes= 2) #to_categorical是用于将类向量（整数）转换为二进制类矩阵,np_utils是keras.utils的一个模块，to_categorical是keras.utils.np_utils模块中的一个函数，用于将类别向量转换为二进制（只有0和1）的矩阵类型表示。
        yield (X_train, Y_train)


if __name__ == "__main__":
    # 模型保存的位置
    log_dir = "./logs/"

    # 打开数据集的txt
    with open(r"./data/dataset.txt","r") as f:
        lines = f.readlines()

    # 打乱行，这个txt主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 90%用于训练，10%用于估计。
    num_val = int(len(lines)*0.1)
    num_train = len(lines) - num_val

    # 建立AlexNet模型
    model = AlexNet()

    # 保存的方式，3世代保存一次
    checkpoint_period1 = ModelCheckpoint(
                                    log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='acc',
                                    save_weights_only=False,
                                    save_best_only=True,
                                    period=3
                                )
    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
                            monitor='acc',
                            factor=0.5,
                            patience=3,
                            verbose=1
                        )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
                            monitor='val_loss',
                            min_delta=0,
                            patience=10,
                            verbose=1
                        )

    # 交叉熵
    model.compile(loss = 'categorical_crossentropy', #用于比较模型的预测和真实的标签，计算损失。
            optimizer = Adam(lr=1e-3), #Adam是一种常用的优化器，lr=1e-3设置了学习率。
            metrics = ['accuracy'])

    # 一次的训练集大小
    batch_size = 128

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    # 开始训练
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size), #从文件中读取图像数据和对应的标签，然后生成一个个的批次用于训练。lines[:num_train]表示训练数据的文件路径列表，batch_size是每个批次的大小。
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
            validation_steps=max(1, num_val//batch_size),
            epochs=50,
            initial_epoch=0,
            callbacks=[checkpoint_period1, reduce_lr])#是ModelCheckpoint用于每隔几个周期保存一次模型，ReduceLROnPlateau用于当准确率不再提高时减小学习率。
    model.save_weights(log_dir+'last2.h5')

