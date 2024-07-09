import math
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import input
DATA_DIR = "Cifar_data/cifar-10-batches-bin"
MAX_STEPS = 4000
EXAMPLES_NUM = 10000

def _InitVarWithWeight(shape, stddev, w=None):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w is not None:
        weightLossse = tf.multiply(tf.nn.l2_loss(var), w)
        tf.add_to_collection("weight_lossse", weightLossse)
    return var

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

class TFNet:
    # 输入数据/标签
    inM = None
    outM = None
    loss = None
    batchSize = 0
    topKOp = None

    # 模型
    modelForward = None
    modelBackward = None

    # tensorflow 会话
    sess = None
    coord = None

    KERNEL_1_SHAPE = [5,5,3,64]
    KERNEL_1_STDDEV = 5e-2
    KERNEL_1_W = 0
    KERNEL_1_STRIDE = [1, 1, 1, 1]
    KERNEL_1_POOL_SHAPE = [1, 3, 3, 1]
    KERNEL_1_POOL_STRIDE = [1, 2, 2, 1]

    KERNEL_2_SHAPE = [5, 5, 64, 64]
    KERNEL_2_STDDEV = 5e-2
    KERNEL_2_W = 0
    KERNEL_2_STRIDE = [1, 1, 1, 1]
    KERNEL_2_POOL_SHAPE = [1, 3, 3, 1]
    KERNEL_2_POOL_STRIDE = [1, 2, 2, 1]

    WEIGHT_1_SHAPE = [-1, 384]
    WEIGHT_1_STDDEV4 = 0.04
    WEIGHT_1_W = 0.04

    WEIGHT_2_SHAPE = [384, 192]
    WEIGHT_2_STDDEV4 = 0.04
    WEIGHT_2_W = 0.04

    WEIGHT_3_SHAPE = [192, 10]
    WEIGHT_3_STDDEV4 = 1 / 192.0
    WEIGHT_3_W = 0

    # 私有函数
    def __init__(self, batchSize, inCnt, learningRate):
        self.batchSize = batchSize
        self.inM = tf.placeholder(tf.float32, [batchSize, inCnt, inCnt, 3])
        self.outM = tf.placeholder(tf.int32, [batchSize])

        # 设置计算图
        # 正向传播
        # 两层卷积
        # 第一层卷积
        kernel1 = _InitVarWithWeight(self.KERNEL_1_SHAPE, self.KERNEL_1_STDDEV, self.KERNEL_1_W)
        kernel1B = tf.Variable(tf.random.uniform([self.KERNEL_1_SHAPE[3]], minval=-0.5, maxval=0.5, dtype=tf.float32))
        conv1 = tf.nn.conv2d(self.inM, kernel1, strides=self.KERNEL_1_STRIDE, padding="SAME")
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, kernel1B))
        pool1 = tf.nn.max_pool(relu1, ksize=self.KERNEL_1_POOL_SHAPE, strides=self.KERNEL_1_POOL_STRIDE, padding="SAME")

        kernel2 = _InitVarWithWeight(self.KERNEL_2_SHAPE, self.KERNEL_2_STDDEV, self.KERNEL_2_W)
        kernel2B = tf.Variable(tf.random.uniform([self.KERNEL_2_SHAPE[3]], minval=-0.5, maxval=0.5, dtype=tf.float32))
        conv2 = tf.nn.conv2d(pool1, kernel2, strides=self.KERNEL_2_STRIDE, padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, kernel2B))
        pool2 = tf.nn.max_pool(relu2, ksize=self.KERNEL_2_POOL_SHAPE, strides=self.KERNEL_2_POOL_STRIDE, padding="SAME")

        # 三层全连接
        # 第一层全连接
        pool2Reshape = tf.reshape(pool2, [batchSize, -1])
        dim = pool2Reshape.get_shape()[1].value
        w1 = _InitVarWithWeight([dim, self.WEIGHT_1_SHAPE[1]], self.KERNEL_1_STDDEV, self.WEIGHT_1_W)
        w1B = tf.Variable(tf.random.uniform([self.WEIGHT_1_SHAPE[1]], minval=-0.5, maxval=0.5, dtype=tf.float32))
        fc1 = tf.nn.relu(tf.matmul(pool2Reshape, w1) + w1B)

        # 第二层全连接
        w2 = _InitVarWithWeight(self.WEIGHT_2_SHAPE, self.WEIGHT_2_STDDEV4, self.WEIGHT_2_W)
        w2B = tf.Variable(tf.random.uniform([self.WEIGHT_2_SHAPE[1]], minval=-0.5, maxval=0.5, dtype=tf.float32))
        fc2 = tf.nn.relu(tf.matmul(fc1, w2) + w2B)

        # 第三等全连接
        w3 = _InitVarWithWeight(self.WEIGHT_3_SHAPE, self.WEIGHT_3_STDDEV4, self.WEIGHT_3_W)
        w3B = tf.Variable(tf.random.uniform([self.WEIGHT_3_SHAPE[1]], minval=-0.5, maxval=0.5, dtype=tf.float32))
        self.modelForward = tf.nn.relu(tf.matmul(fc2, w3) + w3B)
        self.topKOp = tf.nn.in_top_k(self.modelForward, self.outM, 1)

        # 反向传播
        crossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.modelForward,labels=tf.cast(self.outM, tf.int64))
        weightLosses = tf.add_n(tf.get_collection("weight_lossse"))
        self.loss = tf.reduce_mean(crossEntropy) + weightLosses
        self.modelBackward = tf.train.RMSPropOptimizer(learningRate).minimize(self.loss)

    # -------- 导出函数 ---------
    # 训练
    def Train(self, xTrainList, yTrainList):
        # 初始化
        self.coord = tf.train.Coordinator()
        self.sess = tf.Session()
        tf.global_variables_initializer().run(session=self.sess)
        # 创建队列读取数据
        tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

        try:
            # 每隔 100 步打印
            # for step in range(200):
            for step in range(MAX_STEPS):
                start_time = time.time()
                xBatch, yBatch = self.sess.run([xTrainList, yTrainList])
                _, loss_value = self.sess.run([self.modelBackward, self.loss], feed_dict={self.inM: xBatch, self.outM: yBatch})
                duration = time.time() - start_time

                if step % 100 == 0:
                    eps = self.batchSize / duration
                    spb = float(duration)
                    print("step %d, loss=%.2f(%.1f examples/sec; %.3f sec/batch)" % (step, loss_value, eps, spb))
        except Exception as e:
            self.coord.request_stop(e)
        finally:
            self.coord.request_stop()

    # 预测
    def Predict(self, x, y):
        acc = 0
        n = int(math.ceil(EXAMPLES_NUM / self.batchSize))  # math.ceil()函数用于求整
        total = n * self.batchSize

        try:
            # 在一个for循环里面统计所有预测正确的样例个数
            for j in range(n):
                xBatch, yBatch = self.sess.run([x, y])
                predictions = self.sess.run([self.topKOp], feed_dict={self.inM: xBatch, self.outM: yBatch})
                acc += np.sum(predictions)
        except Exception as e:
            self.coord.request_stop(e)
        finally:
            self.coord.request_stop()

        # 打印正确率信息
        print("accuracy = %.3f%%" % ((acc / total) * 100))

    def end(self):
        self.coord.request_stop()
        self.sess.close()

if __name__ == "__main__":
    xTrain, yTrain = input.inputs(data_dir=DATA_DIR, batch_size=100, distorted=True)
    xTest, yTest = input.inputs(data_dir=DATA_DIR, batch_size=100, distorted=None)
    myModel = TFNet(100, 24, 0.1)

    # 训练数据
    myModel.Train(xTrain, yTrain)

    # 推理预测
    myModel.Predict(xTest, yTest)

    # 关闭会话
    myModel.end()