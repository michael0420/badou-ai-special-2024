from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils
from keras.optimizers import Adam
from AlexNetModel import AlexNet
import numpy as np
import cv2
from keras import backend as K
# K.set_image_dim_ordering('tf')
K.set_image_data_format('channels_last')

def data_read(lines,img_dir,batch_size):
    """
    读取数据
    :param lines: 图片对应思维标签数据
    :param img_dir: 图片所在文件夹
    :param batch_size: 批大小
    :return: 输入数据和标签数据
    """
    i = 0
    X_data = []
    Y_data = []
    # 获取一个batch_size大小的数据
    for b in range(batch_size):
        print('b is :',b)
        print(f'i is:{i}\n')
        print('lines[i] is :',lines[i])
        name = lines[i].split(';')[0]
        # 从文件中读取图像
        img = cv2.imread(img_dir + '/' + name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255
        X_data.append(img)
        Y_data.append(lines[i].split(';')[1])
        # break
        # 读完一个周期后重新开始
        i = (i + 1) % n
    # 处理图像
    X_data = resize_image(X_data,(224,224))
    X_data = X_data.reshape(-1, 224, 224, 3)
    Y_data = np_utils.to_categorical(np.array(Y_data), num_classes=2)

    yield (X_data,Y_data)

def resize_image(image, size):
    """
    图片缩放
    :param image: 图片集
    :param size: 缩放后大小
    :return:
    """
    images = []
    for i in image:
        i = cv2.resize(i, size)
        images.append(i)
    images = np.array(images)
    return images

if __name__=='__main__':
    # 设置模型运行方式：train表示训练模型，predict表示模型预测，不进行训练。
    train_mode='train'
    model = AlexNet()
    if train_mode=='train':
        # 数据处理

        dataset_dir=r".\data\dataset.txt"
        img_dir=r".\data\image\train"
        batch_size=1
        with open(dataset_dir, "r") as f:
            lines = f.readlines()
        n = len(lines)
        # np.random.shuffle(lines)
        num_val = int(n * 0.1)
        num_train = n - num_val

        log_dir = "./logs/"   # 模型保存的位置
        # 参数设置
        checkpoint_period1 = ModelCheckpoint(
            log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
            monitor='acc',
            save_weights_only=False,
            save_best_only=True,
            period=3
        )
        # 学习率下降的方式
        reduce_lr = ReduceLROnPlateau(
            monitor='acc',
            factor=0.5,
            patience=3,
            verbose=1
        )
        # 是否需要早停
        early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=10,
            verbose=1
        )

        # 交叉熵
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=1e-3),
                      metrics=['accuracy'])

        # 一次的训练集大小
        batch_size = 3

        print('开始训练...')

        # 开始训练
        model.fit_generator(data_read(lines[:num_train],img_dir,batch_size),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=data_read(lines[num_train:],img_dir,batch_size),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=5,
                            initial_epoch=0,
                            callbacks=[checkpoint_period1, reduce_lr])
        model.save_weights(log_dir + 'test.h5')
        print('训练结束...')
    elif train_mode=='predict':
        model.load_weights("logs/last1.h5")
        img = cv2.imread("./Test.jpg")
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_nor = img_RGB / 255
        img_nor = np.expand_dims(img_nor, axis=0)
        img_resize = cv2.resize(img_nor, (224, 224))
        img_result=np.argmax(model.predict(img_resize))
        print('result image is:',img_result)
        cv2.imshow("test", img)
        cv2.waitKey(0)