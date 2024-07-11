'''
@author linrenkang
基于 keras 实现深度学习框架
'''

from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

class Model:
    model = None

    # 私有函数
    def __init__(self):
        pass

    # 标准化和重塑输入节点数据
    def _XReshape(self, test):
        return test.reshape(test.shape[0], test.shape[1] * test.shape[2]).astype('float32') / 255

    # 输出化图像数据
    def _InitXDataByKeras(self, xTainList):
        return self._XReshape(xTainList)

    # 初始化标签数据
    def _InitYDataByKeras(self, yTrainList):
        return to_categorical(yTrainList, 10)

    # 检测预测的准确率
    def _CheckPredAcc(self, yTrain, yTest):
        acc = 0
        for i in range(len(yTrain)):
            acc += 1 if yTrain[i] == yTest[i] else 0
        return acc / len(yTrain)

    # 导出函数
    # 训练
    def KerasTrain(self, xTainList, yTrainList):
        x = self._InitXDataByKeras(xTainList)
        y = self._InitYDataByKeras(yTrainList)
        # 创建模型
        self.model = models.Sequential()
        # 定义网络层
        self.model.add(layers.Dense(512, activation='relu', input_shape=(x.shape[1],)))
        # 定义输出层
        self.model.add(layers.Dense(10, activation='softmax'))
        # 编译网络层
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        # 训练
        self.model.fit(x=x, y=y, epochs=5, batch_size=128, verbose=2)
        # 评估模型
        loss, acc = self.model.evaluate(x, y, verbose=1)
        print("loss: ", loss, " acc: ", acc)

    # 推理预测
    def KerasPredict(self, xTestList, yTestList):
        x = self._InitXDataByKeras(xTestList)
        yTrain = self.model.predict_classes(x)
        print("Predict acc: ", self._CheckPredAcc(yTrain, yTestList))

'''
    基于 keras 调用 
'''
if __name__ == "__main__":
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    myModel = Model()

    # 训练数据
    myModel.KerasTrain(xTrain, yTrain)
    # 推理预测
    myModel.KerasPredict(xTest, yTest)































