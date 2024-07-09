# Author: Zhenfei Lu
# Created Date: 6/25/2024
# Version: 1.0
# Email contact: luzhenfei_2017@163.com, zhenfeil@usc.edu

from NN.Layers import *
from NN.LossFunctions import *
from NN.LossObject import *
import random
import matplotlib.pyplot as plt

class NeuralNetWorks(object):
    def __init__(self):
        self.Layers = list()

    def __call__(self, *args, **kwargs):
        if len(kwargs) != 0:
            return self.fwd(**kwargs)
        elif len(args) != 0:
            return self.fwd(*args)

    def addInputLayer(self, inputNum):
        self.Layers.append(InputLayer(inputNum))

    def addFCLayer(self, neuronsNum, activation):
        previousLayerNeuronsNum = self.Layers[-1].neuronsNum
        self.Layers.append(FullyConnectedLayer(neuronsNum, activation, previousLayerNeuronsNum))

    def setLossFunction(self, LossFunctionType):
        if(LossFunctionType == LossFuncType.MSE):
            self.LossFunction = MSE()
        elif(LossFunctionType == LossFuncType.CROSSENTROPY):
            self.LossFunction = CROSSENTROPY()
        else:
            print("fatal error: unknown type of loss function")

    def calLoss(self, y_predict, y):
        loss_total = 0
        N_y = y.shape[0]
        for j in range(0, N_y):
            loss_total += self.LossFunction.func(y_predict, y).item()
        loss_total = loss_total / N_y
        return LossObject(self, loss_total, y)

    def calAccuracy(self, x, y, printResult=False, plotResult=False):
        acc = 0
        N_y = y.shape[0]
        for j in range(0, N_y):
            training_x = x[j].reshape(-1, 1)
            imageX_flattenN = training_x.shape[0]
            training_y = y[j].reshape(-1, 1)
            # onehotY_N = training_y.shape[0]
            y_predict = self.fwd(training_x)
            y_predict_index = int(np.argmax(y_predict, axis=0))
            y_truth_index = int(np.argmax(training_y, axis=0))
            if(y_predict_index == y_truth_index):
                acc = acc + 1
                if (printResult):
                    print(f"predict: {y_predict_index}, truth: {y_truth_index}, result: Correct")
                if (plotResult):
                    plt.figure()
                    plt.title(f"predict: {y_predict_index}, truth: {y_truth_index}, result: Correct")
                    h_w = int(np.sqrt(imageX_flattenN))
                    plt.imshow(training_x.reshape(h_w, h_w), cmap='gray')  # matplot lib only accepts RGB order image
            else:
                if (printResult):
                    print(f"predict: {y_predict_index}, truth: {y_truth_index}, result: Wrong")
                if (plotResult):
                    plt.figure()
                    plt.title(f"predict: {y_predict_index}, truth: {y_truth_index}, result: Wrong")
                    h_w = int(np.sqrt(imageX_flattenN))
                    plt.imshow(training_x.reshape(h_w, h_w), cmap='gray')  # matplot lib only accepts RGB order image
        return acc/N_y

    def fwd(self, X):
        if (len(self.Layers) < 1):
            print("No input layer")
            return None
        if(len(self.Layers) < 2):
            print("Warning: only has one inputLayer, no hidden layers or output layers. NN does not need to train")

        currentLayer = self.Layers[0]
        currentLayer.setInput(X)
        for i in range(1, len(self.Layers)):
            previousLayer = self.Layers[i-1]
            currentLayer = self.Layers[i]
            currentLayer.updateOutputs(previousLayer.outputs)
        return currentLayer.outputs

    def bwd(self, yGroundTruth):
        currentLayer = self.Layers[-1]
        currentLayer.updateError(nextLayerError=None, nextLayerWeights=None, isHiddenLayer=False, outputsGroundTruth=yGroundTruth, lossFunction=self.LossFunction)
        currentLayer.updateGradients(previousLayerOutput=self.Layers[-2].outputs)
        for i in range(len(self.Layers)-2, 0, -1):
            nextLayer = self.Layers[i+1]
            currentLayer = self.Layers[i]
            previousLayer = self.Layers[i-1]
            currentLayer.updateError(nextLayerError=nextLayer.error, nextLayerWeights=nextLayer.weights, isHiddenLayer=True, outputsGroundTruth=None, lossFunction=None)
            currentLayer.updateGradients(previousLayerOutput=previousLayer.outputs)

    def applyGradient(self, learningRate, batchSize, optimizerType):
        for i in range(1, len(self.Layers)):
            currentLayer = self.Layers[i]
            currentLayer.applyGradients4DerivableParams(learningRate, batchSize, optimizerType)

    def clearGradient(self):
        for i in range(1, len(self.Layers)):
            currentLayer = self.Layers[i]
            currentLayer.clearGradients()

    def data_iter(self, batch_size, features, labels, shuffle=True):
        num_examples = len(features)
        indices = list(range(num_examples))
        if(shuffle):
            random.shuffle(indices)
        for i in range(0, num_examples, batch_size):
            batch_indices = indices[i: min(i + batch_size, num_examples)]
            yield features[batch_indices], labels[batch_indices]

    def fit(self, x, y, learningRate, epoches, batchSize, shuffle=True, optimizerType=OptimizerType.ADAM, LossFunctionType=LossFuncType.MSE, metric="loss"):
        N_y = y.shape[0]
        N_x = x.shape[0]
        if(N_y != N_x):
            print("Fatal error: inputdata num is not equals to the true outputdata num!!!")
            return
        self.setLossFunction(LossFunctionType)
        avg_loss = []
        accuracy = []
        for i in range(0, epoches):
            for X, Y in self.data_iter(batchSize, x, y, shuffle):
                self.clearGradient()
                for j in range(0, batchSize):
                    training_x = X[j].reshape(-1,1)
                    training_y = Y[j].reshape(-1,1)
                    loss = self.calLoss(self.fwd(training_x), training_y)
                    loss.bwd()
                self.applyGradient(learningRate, N_y, optimizerType)
            loss_total = self.calLoss(self.fwd(training_x), training_y)
            avg_loss.append(loss_total())
            if(metric == "accuracy"):
                acc = self.calAccuracy(x, y)
                accuracy.append(acc)
                print("epoch = " + str(i) + ", LossFunction = " + str(self.LossFunction.LossFuncType) + ", AvgLoss = " + str(loss_total()) + ", Accuracy = " + str(acc))
            else:
                print("epoch = " + str(i) + ", LossFunction = " + str(self.LossFunction.LossFuncType) + ", AvgLoss = " + str(loss_total()))
        return {"loss": avg_loss, "accuracy": accuracy}

    def plotMetrics(self, loss_list, acc_list):
        if(len(loss_list) > 0):
            plt.figure()
            plt.title("Training loss")
            plt.plot(loss_list)
        if(len(acc_list) > 0):
            plt.figure()
            plt.title("Training accuracy")
            plt.plot(acc_list)
