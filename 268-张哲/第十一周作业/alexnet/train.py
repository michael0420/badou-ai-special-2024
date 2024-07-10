from keras.callbacks import TensorBoard,ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from keras.utils import np_utils
from keras.optimizers import Adam
from model.AlexNet import AlexNet
import numpy as np
import utils
import cv2
from keras import backend as K
# K.set_image_dim_ordering('tf')

def generate_arrays_from_file(lines,batch_size):
    # 获取总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # 从文件中读取图像
            img = cv2.imread(r".\data\image\train" + '/' + name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255
            X_train.append(img)
            Y_train.append(lines[i].split(';')[1])
            # 读完一个周期后重新开始
            i = (i + 1) % n
        # 处理图像
        X_train = utils.resize_img(X_train, (224, 224))
        X_train = X_train.reshape(-1, 224, 224, 3)
        Y_train = np_utils.to_categorical(np.array(Y_train), num_classes=2)
        yield (X_train, Y_train)

if __name__ == "__main__":
    # 模型保存的位置
    log_dir = "./logs/"

    # 打开数据集的txt
    with open(r".\data\dataset.txt", "r") as f:
        lines = f.readlines()

    # 打乱行，这个txt主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 90%用于训练，10%用于估计。
    num_val = int(len(lines) * 0.1)
    num_train = len(lines) - num_val

    # 建立AlexNet模型
    model = AlexNet()

    # 保存的方式，3世代保存一次
    '''1.filename：字符串，保存模型的路径，filepath可以是格式化的字符串，里面的占位符将会被epoch值和传入on_epoch_end的logs关键字所填入。
例如：filepath = “weights_{epoch:03d}-{val_loss:.4f}.h5”，则会生成对应epoch和测试集loss的多个文件。

2.monitor：需要监视的值，通常为：val_acc 、 val_loss 、 acc 、 loss四种。

3.verbose：信息展示模式，0或1。为1表示输出epoch模型保存信息，默认为0表示不输出该信息。

4.save_best_only：当设置为True时，将只保存在测试集上性能最好的模型。

5.mode：‘auto’，‘min’，‘max’之一，在save_best_only=True时决定性能最佳模型的评判准则，例如，当监测值为val_acc时，模式应为max，当检测值为val_loss时，模式应为min。在auto模式下，评价准则由被监测值的名字自动推断。

6.save_weights_only：若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）。

7.period：CheckPoint之间的间隔的epoch数。
'''
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
        factor=0.5,  # 触发条件后lr*=factor
        patience=3,
        verbose=1  # 触发条件后print
    )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    '''1.monitor: 监控的数据接口，有’acc’,’val_acc’,’loss’,’val_loss’等等。正常情况下如果有验证集，就用’val_acc’或者’val_loss’。

2.mode: 就’auto’, ‘min’, ‘,max’三个可能。如果知道是要上升还是下降，建议设置一下。例如监控的是’acc’，那么就设置为’max’。

3.min_delta：增大或减小的阈值，只有大于这个部分才算作改善（监控的数据不同，变大变小就不确定）。这个值的大小取决于monitor，也反映了你的容忍程度。

4.patience：能够容忍多少个epoch内都没有改善。patience的大小和learning rate直接相关。在learning rate设定的情况下，前期先训练几次观察抖动的epoch number，patience设置的值应当稍大于epoch number。在learning rate变化的情况下，建议要略小于最大的抖动epoch number。

5.baseline：监控数据的基线值，如果在训练过程中，模型训练结果相比于基线值没有什么改善的话，就停止训练
'''
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1
    )
    # 交叉熵
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.01),
                  metrics=['accuracy']
                  )
    batch_size = 128
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    # 开始训练
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
            validation_steps=max(1, num_val//batch_size),
            epochs=5,
            initial_epoch=0,
            callbacks=[checkpoint_period1, reduce_lr])
    model.save_weights(log_dir + 'last1.h5')