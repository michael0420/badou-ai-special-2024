'''
@author linrenkang
基于 keras 训练神经网络
'''
import keras.optimizers
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.python.client import device_lib
import network
import utils
import cv2
import os

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

class Model:
    network = None
    LOG_DIR = "./logs/"

    # 私有函数
    def __init__(self):
        pass

    # 处理数据
    def _InitData(self, x, y):
        x = utils.ResizeImage(x, (utils.INPUT_SHAPE[0], utils.INPUT_SHAPE[1]))
        x = x.reshape(-1, utils.INPUT_SHAPE[0], utils.INPUT_SHAPE[1], utils.INPUT_SHAPE[2])
        y = keras.utils.to_categorical(np.array(y), num_classes=utils.OUTPUT_SHAPE)
        return x, y

    # 遍历函数
    def _Iterator(self, namePath, imgPath, batchSize):
        while True:
            cnt = 0
            with open(namePath) as f:
                i, x, y = 0, [], []
                for line in f:
                    name = line.split(';')[0]
                    lable = line.split(';')[1]

                    img = cv2.imread(imgPath + '/' + name)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img / 255
                    x.append(img)
                    y.append(lable)

                    cnt += 1
                    i = (i + 1) % batchSize
                    if i == 0:
                        yield self._InitData(x, y)
                        x.clear()
                        y.clear()
                yield self._InitData(x, y)
                x.clear()
                y.clear()

    def _InitGPU(self):
        os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
        os.environ["CL_GPUOFFSET"] = "1"

        # 配置Keras使用OpenCL后端
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        set_session(sess)

        # 创建回调以在训练过程中使用OpenCL设备
        # callbacks = [OpenCLDeviceCallback(queue)]

        os.environ["CUDA_VISIBLD_DEVICES"] = "0"

        print("GPU available： ", tf.test.is_gpu_available())
        config = tf.ConfigProto()
        # Best-fit with coalescing
        config.gpu_options.allocator_type = 'BFC'
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        K.set_session(session)

        gpus = K.tensorflow_backend._get_available_gpus()
        if gpus:
            print(f"Keras is using GPU: ", gpus)
        else:
            print("Keras is not using GPU.")

    # 使用 NPU 加速
    def _InitByARMNPU(self):
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
        keras.backend.set_session(sess)

    # 导出函数
    # 训练
    def Train(self, namePath, imgPath, testNamePath, testImgPath, rate, epoches, batchSize, trainNum, testNum):
        print('with batch size {}.'.format(batchSize))
        self._InitByARMNPU()
        self.network = network.Network()

        # 3代保存一次
        checkpoint_period1 = ModelCheckpoint(
            self.LOG_DIR + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
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
            # verbose=1
        )
        # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
        early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=10,
            verbose=1
        )

        # 交叉熵
        self.network.net.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.RMSprop(lr=rate),
                      metrics=['accuracy'])

        # 开始训练
        self.network.net.fit_generator(self._Iterator(namePath, imgPath, batchSize),
                               steps_per_epoch=max(1, trainNum),
                               validation_data=self._Iterator(testNamePath, testImgPath, batchSize),
                               validation_steps=max(1, testNum),
                               epochs=epoches,
                               initial_epoch=0,
                               callbacks=[checkpoint_period1, reduce_lr])

        # 保存权重值
        self.network.net.save_weights(self.LOG_DIR + 'last1.h5')

    # 推理预测
    def Predict(self, imgPath):
        model = self.network.net
        imgResize = utils.InitInput(imgPath)
        utils.PrintAnswer(np.argmax(model.predict(imgResize)))

'''
    基于 keras 调用 
'''
if __name__ == "__main__":
    comPath = r".\data"
    trainNamePath = comPath + r"\dataset.txt"
    trainImgPath = comPath + r"\image\train"
    testNameTestPath = comPath + r"\dataset_test.txt"
    trainImgTestPath = comPath + r"\image\train_test"
    testImg = r".\data\Test.jpg"

    with open(trainNamePath, "r") as f:
        lines = f.readlines()
    with open(testNameTestPath, "r") as f:
        linesTest = f.readlines()

    myModel = Model()
    batchSize = 50
    trainBum = int(len(lines) / batchSize)
    testNum = int(len(linesTest) / batchSize)
    myModel.Train(trainNamePath, trainImgPath, testNameTestPath, trainImgTestPath,
                  1e-3, 5, batchSize, trainBum, testNum)
    myModel.Predict(testImg)
