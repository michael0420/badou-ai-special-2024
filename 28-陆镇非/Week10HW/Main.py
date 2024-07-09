# Author: Zhenfei Lu
# Created Date: 6/25/2024
# Version: 1.0
# Email contact: luzhenfei_2017@163.com, zhenfeil@usc.edu

from NN.NeuralNetWorks import *
from NN.Activations import *
from NN.LossFunctions import *
import numpy as np
import matplotlib.pyplot as plt

class Solution(object):
    def __init__(self):
        self.regression_test()
        self.classification_test()
        plt.show()

    def regression_test(self):
        X = np.linspace(-5, 5, 100).reshape(-1, 1)
        Y = 1 * np.sin(X)

        nn = NeuralNetWorks()
        nn.addInputLayer(1)
        for i in range(0, 3):
            nn.addFCLayer(64, TANH())
        # nn.addFCLayer(64, RELU())
        # nn.addFCLayer(64, SIGMOID())
        # nn.addFCLayer(64, TANH())
        nn.addFCLayer(1, NONEACTIVATION())

        hist = nn.fit(X, Y, learningRate=0.0001, epoches=3000, batchSize=10, shuffle=True,
                      optimizerType=OptimizerType.ADAM,LossFunctionType=LossFuncType.MSE, metric="loss")
        nn.plotMetrics(hist["loss"], hist["accuracy"])

        Y_predict = []
        for i in range(0, X.shape[0]):
            Y_predict.append(nn(X[i].reshape(-1, 1)).item())

        plt.figure()
        plt.plot(X, np.array(Y_predict).reshape(-1, 1), label='predict')
        plt.plot(X, Y,  label='truth')
        plt.legend()
        plt.title('predict vs truth')

    def classification_test(self):
        def get_data(filePath):
            training_data_file = open(filePath)
            trainning_data_list = training_data_file.readlines()
            training_data_file.close()
            training_X = []
            training_Y = []
            for record in trainning_data_list:
                all_values = record.split(',')
                x = (np.asfarray(all_values[1:])) / 255.0
                training_X.append(x)
                y = np.zeros((10,))  # one-hot vector
                y[int(all_values[0])] = 1
                training_Y.append(y)
            training_X = np.array(training_X)
            training_Y = np.array(training_Y)
            return training_X, training_Y

        training_X, training_Y = get_data("dataset/mnist_train.csv")
        test_X, test_Y = get_data("dataset/mnist_test.csv")

        nn = NeuralNetWorks()
        nn.addInputLayer(training_X.shape[1])   # heigth*width input  28*28
        for i in range(0, 3):
            nn.addFCLayer(256, SIGMOID())
        nn.addFCLayer(training_Y.shape[1], SOFTMAX())  # onehot output

        hist = nn.fit(x=training_X, y=training_Y, learningRate=0.001, epoches=300, batchSize=10, shuffle=True,
                      optimizerType=OptimizerType.ADAM, LossFunctionType=LossFuncType.CROSSENTROPY, metric="accuracy")
        nn.plotMetrics(hist["loss"], hist["accuracy"])

        # evaluate
        acc = nn.calAccuracy(test_X, test_Y, printResult=True, plotResult=True)
        print("accuracy = " + str(acc))


if __name__ == "__main__":
    solution = Solution()


