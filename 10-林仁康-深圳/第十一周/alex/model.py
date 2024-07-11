'''
@author linrenkang
基于 keras Alex 神经网络
'''
import keras.optimizers
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import network
import utils
import cv2

class Model:
    net = None
    LOG_DIR = "./logs/"

    # 私有函数
    def __init__(self):
        pass

    # 处理数据
    def _InitData(self, x, y):
        x = utils.resize_image(x, (network.INPUT_SHAPE[0], network.INPUT_SHAPE[1]))
        x = x.reshape(-1, network.INPUT_SHAPE[0], network.INPUT_SHAPE[1], network.INPUT_SHAPE[2])
        y = keras.utils.to_categorical(np.array(y), num_classes=2)
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

    # 导出函数
    # 训练
    def Train(self, namePath, imgPath, testNamePath, testImgPath, rate, epoches, batchSize, trainNum, testNum):
        print('with batch size {}.'.format(batchSize))
        self.net = network.network().net

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
        self.net.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.RMSprop(lr=rate),
                      metrics=['accuracy'])

        # 开始训练
        self.net.fit_generator(self._Iterator(namePath, imgPath, batchSize),
                               steps_per_epoch=max(1, trainNum),
                               validation_data=self._Iterator(testNamePath, testImgPath, batchSize),
                               validation_steps=max(1, testNum),
                               epochs=epoches,
                               initial_epoch=0,
                               callbacks=[checkpoint_period1, reduce_lr])

        # 保存权重值
        self.net.save_weights(self.LOG_DIR + 'last1.h5')

    # 推理预测
    def Predict(self, imgPath):
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255
        img = np.expand_dims(img, axis=0)
        img_resize = utils.resize_image(img, (224, 224))
        res = np.argmax(self.net.predict(img_resize))
        print(utils.print_answer(res))

'''
    基于 keras 调用 
'''
if __name__ == "__main__":
    with open(r".\data\dataset.txt", "r") as f:
        lines = f.readlines()
    with open(r".\data\dataset_test.txt", "r") as f:
        linesTest = f.readlines()

    myModel = Model()
    batchSize = 128
    trainBum = int(len(lines) / batchSize)
    testNum = int(len(linesTest) / batchSize)
    myModel.Train(r".\data\dataset.txt", r".\data\image\train",
                  r".\data\dataset_test.txt", r".\data\image\train_test",
                  1e-3, 5, batchSize, trainBum, testNum)
    myModel.Predict(r".\data\Test.jpg")
