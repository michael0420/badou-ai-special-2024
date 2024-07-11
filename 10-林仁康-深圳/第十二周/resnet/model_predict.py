'''
@author linrenkang
神经网络预测
'''
import numpy as np
import network
import utils

class Model:
    net = None

    # 私有函数
    def __init__(self):
        pass

    # 推理预测
    def Predict(self, trainFile, imgPath):
        model = network.Network().net
        model.load_weights(trainFile)
        imgInit = utils.InitInput(imgPath)
        utils.PrintAnswer(np.argmax(model.predict(imgInit)))

'''
    基于 keras 调用 
'''
if __name__ == "__main__":
    comPath = r".\data"
    testImg = comPath + r"\Test.jpg"
    trainFile = "./logs/last1.h5"

    myModel = Model()
    myModel.Predict(trainFile, testImg)
