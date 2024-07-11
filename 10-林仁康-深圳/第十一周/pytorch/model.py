import numpy as np
import torch
from tensorflow.keras.datasets import mnist
import net

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

# 计算分类
def _ToCategories(res):
    categories = []
    for r in res:
        categories.append(np.argmax(r))
    return np.array(categories)

# 检验正确性
def _CheckPredAcc(yPred, yTest):
    acc = 0
    for i in range(len(yPred)):
        acc += 1 if yPred[i] == yTest[i] else 0
    return acc / len(yPred)

# 自定义模型
class Model:
    network = None
    costModel = None
    optimizer = None

    # ---------- 私有函数 --------
    def __init__(self, network, rate):
        self.network = network

        self.costModel = torch.nn.CrossEntropyLoss()
        # costModel = torch.nn.MSELoss()

        self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr=rate)
        # self.optimizer = torch.optim.Adam(self.network.parameters(), lr=rate)
        # self.optimizer = torch.optim.SGD(self.network.parameters(), lr=rate)

    # 批训练
    def TrainByBatch(self, inM, outM, size):
        for i in range(size):
            x = torch.tensor(np.array(inM[i], ndmin=2), requires_grad=True)
            y = torch.tensor(np.array(outM[i], ndmin=2), requires_grad=True)

            # 初始化梯度
            self.optimizer.zero_grad()
            # 正向传播
            outputs = self.network(x)
            # 计算损失
            loss = self.costModel(outputs, y)
            # 反向传播
            loss.backward()
            # 更新权重
            self.optimizer.step()
            # print("train loss: ", loss.item())

    # ------------ 导出函数 -----------
    # 训练数据
    def Train(self, xTrainList, yTrainList, epochs, batchSize):
        for i in range(epochs):
            total = len(xTrainList)
            for batch in range(0, total, batchSize):
                size = batchSize if batch + batchSize < total else total - batch
                inM = xTrainList[batch:batch + size]
                outM = yTrainList[batch:batch + size]
                self.TrainByBatch(inM, outM, size)
                yPred = myModel.Predict(inM)
                yPred = _ToCategories(yPred)
                yTest = _ToCategories(outM)
                print("batch", batch, "/", total)
                print("train acc: ", _CheckPredAcc(yPred, yTest))
            print("epoch", i + 1, "/", epochs)

    # 预测数据
    def Predict(self, xTest):
        inM = torch.tensor(xTest)
        outputs = self.network(inM)
        outM = np.array(torch.asarray(outputs))
        return outM

if __name__ == "__main__":
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    # 创建神经网路
    net = net.Net(xTrain.shape[1] * xTrain.shape[2], 512, 10)
    # 创建模型
    myModel = Model(net, 0.01)

    # 训练数据
    x = _XReshape(xTrain)
    y = _YReshape(yTrain, 10)
    myModel.Train(x, y, 5, 128)

    # 推理预测
    x = _XReshape(xTest)
    yPred = myModel.Predict(x)
    yPred = _ToCategories(yPred)
    print("test acc: ", _CheckPredAcc(yPred, yTest))


