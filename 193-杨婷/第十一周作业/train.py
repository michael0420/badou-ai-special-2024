from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils
from keras.optimizers import Adam
from model.AlexNet import AlexNet
import numpy as np
import utils
import cv2
from keras import backend as K

# K.set_image_dim_ordering('tf')  # hwc


def generate_arrays_from_file(lines, batch_size):
    n = len(lines)
    # print(n)
    i = 0
    while 1:  # 创建一个无限循环，以持续生成批次数据
        X_train = []
        Y_train = []

        # 获取一个batch_size大小的数据
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)   # 如果这是批次中的第一个元素，先随机打乱文件行，以实现数据的随机性
            name = lines[i].split(';')[0]  # 把图片文件名分割出来
            img = cv2.imread(r'.\data\image\train'+'/'+name)  # 从文件中读取图像
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img/255
            X_train.append(img)
            Y_train.append(lines[i].split(';')[1])
            # print(X_train)
            # print(Y_train)
            # 读完一个周期后重新开始
            i = (i+1) % n
            # print(i)
        X_train = utils.resize_image(X_train, (224, 224))
        X_train = X_train.reshape(-1, 224, 224, 3)
        Y_train = np_utils.to_categorical(np.array(Y_train), num_classes=2)  # 将Y_train中的标签转换为独热编码格式，假设有2个类别
        yield X_train, Y_train


if __name__ == "__main__":
    # 模型保存的位置
    log_dir = "./logs/"
    with open(r".\data\dataset.txt", "r") as f:
        lines = f.readlines()

    np.random.seed(1)
    np.random.shuffle(lines)
    np.random.seed(None)  # 后面不需要固定随机数种子，用None取消

    # 90%用于训练，10%用于估计。
    num_val = int(len(lines)*0.1)
    num_train = len(lines)-num_val

    # 建立AlexNet模型
    model = AlexNet()

    # 保存的方式，3世代保存一次
    checkpoint_period1 = ModelCheckpoint(
        filepath=log_dir+'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='acc',  # 指定了要监控的量
        save_weights_only=False,
        save_best_only=True,  # 只有当被监控的量在当前轮次中有所改善时，才会保存模型
        period=3  # 还要受到 save_best_only 的影响，只有在这3个轮次中表现最好的时候才会保存
    )

    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
        monitor='acc',
        factor=0.5,  # 当调整发生时，学习率将被乘以这个因子
        patience=3,  # 如果监控的指标在连续3个epoches中没有改善，则触发学习率调整
        verbose=1  # 1 表示输出日志信息
    )

    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,  # 判断指标改善的最小阈值,设置为0意味着即使验证损失有非常小的减少也会被认为是改善
        patience=10,
        verbose=1
    )

    # 交叉熵
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=1e-3),
        metrics=['accuracy']  # 评估指标
    )

    batch_size = 128
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    # 开始训练
    model.fit_generator(
        generate_arrays_from_file(lines[:num_train], batch_size),
        steps_per_epoch=max(1, num_train//batch_size),  # 从生成器中获取的批次数,但至少要为1（以防num_train小于batch_size）
        validation_data=generate_arrays_from_file(lines[num_train:], batch_size),  # 这是验证数据的生成器
        validation_steps=max(1, num_val//batch_size),
        epochs=50,
        initial_epoch=0,
        callbacks=[checkpoint_period1, reduce_lr]  # 指定了一组回调函数，这些函数将在训练的不同阶段被调用
    )
    model.save_weights(log_dir+'last2.h5')










