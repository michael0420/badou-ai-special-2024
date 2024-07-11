import math

import numpy
import numpy as np
import scipy.special
from tensorflow.keras.datasets import mnist

# 标准化和重塑输入节点数据
def _XReshape(x):
    return x.reshape(x.shape[0], x.shape[1] * x.shape[2]).astype('float32') / 255 * 0.99 + 0.01

# 标准化标签数据
def _YReshape(y, categories):
    n = len(y)
    outList = np.zeros((n, categories)) + 0.01
    for i in range(n):
        outList[i][y[i]] = 0.99
    return outList

# 检验正确性
def _CheckPredAcc(yPred, yTest):
    acc = 0
    for i in range(len(yPred)):
        acc += 1 if yPred[i] == yTest[i] else 0
    return acc / len(yPred)

def _InitOneList(initList):
    return [0.01 if x == 0 else x for x in initList]
    # l = [idx for idx, val in enumerate(initList) if val == 0]
    # initList[initList==0] = 0.01

def _InitData(initList):
    if (len(initList.shape) == 1):
        return _InitOneList(initList)

    for row in range(len(initList)):
        initList[row] = _InitOneList(initList[row])
    return initList

class Network:
    # 节点数据
    inNodesCnt = None
    hdNodesCnt = None
    outNodesCnt = None

    # 权重
    weightIH = None
    weightHO = None

    # 偏置
    biasIH = None
    biasHO = None

    # 学习率
    rate = 0

    # 激活函数
    activationFun = None

    # 中间过程
    inM = None
    hdInM = None
    hdOutM = None
    finalInM = None
    finalOutM = None

    # 私有函数
    def __init__(self, inCnt, hideCnt, outCnt, learningRate):
        # 初始化节点
        self.inNodesCnt = inCnt
        self.hdNodesCnt = hideCnt
        self.outNodesCnt = outCnt

        # 初始化权重/偏置(权重在[-0.5, 0.5))
        self.weightIH = np.random.rand(self.hdNodesCnt, self.inNodesCnt) - 0.5
        self.weightHO = np.random.rand(self.outNodesCnt, self.hdNodesCnt) - 0.5
        self.biasIH = np.random.rand(self.hdNodesCnt, 1) - 0.5
        self.biasHO = np.random.rand(self.outNodesCnt, 1) - 0.5

        # self.weightIH = (self.weightIH) * 0.99 + 0.01
        # self.weightHO = (self.weightHO) * 0.99 + 0.01
        # self.biasIH = (self.biasIH) * 0.99 + 0.01
        # self.biasHO = (self.biasHO) * 0.99 + 0.01

        self.weightIH = _InitData(self.weightIH)
        self.weightHO = _InitData(self.weightHO)
        self.biasIH = _InitData(self.biasIH)
        self.biasHO = _InitData(self.biasHO)

        # 初始化学习率
        self.rate = learningRate

        # 设置激活函数
        self.activationFun = lambda x: scipy.special.expit(x)
        # softmax
        self.outFunction = lambda x: np.exp(x) / np.sum(np.exp(x))

    # 正向传播
    def _TrainForward(self, inM):
        # print("_TrainForward inM: ", inM)
        self.inM = inM
        self.hdInM = np.dot(self.weightIH, self.inM.T) + self.biasIH
        self.hdOutM = self.activationFun(self.hdInM)
        self.finalInM = np.dot(self.weightHO, self.hdOutM) + self.biasHO
        self.finalOutM = self.activationFun(self.finalInM)
        # self.finalOutM = self.outFunction(self.finalOutM)
        return self.finalOutM

    # 反向传播
    def _TrainBack(self, yTrainM):
        # 输出层梯度顶
        gradHO = (yTrainM.T - self.finalOutM) * self.finalOutM * (1 - self.finalOutM)
        deltaWHo = np.dot(gradHO, self.hdOutM.T)
        gradIH = np.dot(self.weightHO.T, gradHO) * self.hdOutM * (1 - self.hdOutM)
        deltaWIH = np.dot(gradIH, self.inM)

        self.weightHO += self.rate * deltaWHo
        self.weightIH += self.rate * deltaWIH

    # 标注分类的数字
    def _ToCategories(self, res):
        categories = []
        for r in res:
            categories.append(np.argmax(r))
        return np.array(categories)

    # 按批训练
    def _TrainByBatch(self, inM, outM, size):
        for i in range(size):
            self._TrainForward(np.array(inM[i], ndmin=2))
            self._TrainBack(np.array(outM[i], ndmin=2))
        yPred = self.Predict(np.array(inM, ndmin=2))
        yTest = self._ToCategories(np.array(outM, ndmin=2))
        print("train acc: ", _CheckPredAcc(yPred, yTest))

    # -------- 导出函数 ---------
    # 训练
    def Train(self, xTrainList, yTrainList, epochs, batchSize):
        total = len(xTrainList)
        for i in range(epochs):
            batch = 0
            while batch < total:
                size = batchSize if batch + batchSize < total else total - batch
                inM = xTrainList[batch:batch + size]
                outM = yTrainList[batch:batch + size]
                batch += size
                self._TrainByBatch(inM, outM, size)
                print("batch", batch, "/", total)
            print("epoch", i + 1, "/", epochs)

    # 预测
    def Predict(self, xTrainList):
        inM = np.array(xTrainList, ndmin=2)
        outM = self._TrainForward(inM).T
        return self._ToCategories(outM)

if __name__ == "__main__":
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    myModel = Network(xTrain.shape[1] * xTrain.shape[2], 512, 10, 0.1)

    # 训练数据
    x = _XReshape(xTrain)
    y = _YReshape(yTrain, 10)
    myModel.Train(x, y, 5, 128)

    # 推理预测
    x = _XReshape(xTest)
    yPred = myModel.Predict(x)
    print("test acc: ", _CheckPredAcc(yPred, yTest))
