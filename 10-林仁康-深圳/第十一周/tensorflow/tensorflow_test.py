import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# 标准化和重塑输入节点数据
def _XReshape(xTainList):
    return xTainList.reshape(xTainList.shape[0], xTainList.shape[1] * xTainList.shape[2]).astype('float32') / 255

# 标准化标签数据
def _YReshape(yTainList, categories):
    n = len(yTainList)
    outList = np.zeros((n, categories)) + 0.01
    for i in range(n):
        outList[i][yTainList[i]] = 1
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

class TFNet:
    # 输入数据/标签
    inM = None
    outM = None

    # 模型
    modelForward = None
    modelBackward = None

    # tensorflow 会话
    sess = None

    # 私有函数
    def __init__(self, inCnt, hideCnt, outCnt, learningRate):
        self.inM = tf.placeholder("float", shape=(inCnt, None))
        self.outM = tf.placeholder("float", shape=(outCnt, None))

        # 初始化权重/偏置(权重在[-0.5, 0.5))
        weightIH = tf.Variable(tf.random_normal((hideCnt, inCnt)))
        weightHO = tf.Variable(tf.random_normal((outCnt, hideCnt)))
        biasIH = tf.Variable(tf.random_normal((hideCnt, 1)))
        biasHO = tf.Variable(tf.random_normal((outCnt, 1)))

        # 初始化学习率
        rate = learningRate

        # 设置计算图
        # 正向传播
        hdOutM = tf.nn.sigmoid(tf.matmul(weightIH, self.inM) + biasIH)
        self.modelForward = tf.nn.sigmoid(tf.matmul(weightHO, hdOutM) + biasHO)

        # 反向传播
        crossEntropy = tf.reduce_mean(tf.square(self.outM - self.modelForward))
        self.modelBackward = tf.train.GradientDescentOptimizer(rate).minimize(crossEntropy)

    # 导出函数
    # 训练
    def Train(self, xTrainList, yTrainList, epochs, batchSize):
        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

        total = len(xTrainList)
        for i in range(epochs):
            batch = 0
            while batch < total:
                inM = np.array(xTrainList[batch:batch + batchSize], ndmin=2).astype("float32").T
                outM = np.array(yTrainList[batch:batch + batchSize], ndmin=2).astype("float32").T

                self.sess.run(self.modelBackward, feed_dict={self.inM: inM, self.outM: outM})
                batch += batchSize
                # print("batch", batch, "/", total)
            yPred = self.Predict(xTrainList)
            yPred = _ToCategories(yPred)
            yTest = _ToCategories(yTrainList)
            print("epoch", i, "/", epochs)
            print("train acc: ", _CheckPredAcc(yPred, yTest))

    # 预测
    def Predict(self, xTrainList):
        inM = np.array(xTrainList, ndmin=2).astype("float32").T
        yPred = self.sess.run(self.modelForward, feed_dict={self.inM: inM})
        return yPred.T

    def end(self):
        self.sess.close()

if __name__ == "__main__":
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    myModel = TFNet(xTrain.shape[1] * xTrain.shape[2], 512, 10, 0.1)

    # 训练数据
    x = _XReshape(xTrain)
    y = _YReshape(yTrain, 10)
    myModel.Train(x, y, 100, 128)

    # 推理预测
    x = _XReshape(xTest)
    yPred = myModel.Predict(x)
    yPred = _ToCategories(yPred)
    print("acc: ", _CheckPredAcc(yPred, yTest))

    # 关闭会话
    myModel.end()


