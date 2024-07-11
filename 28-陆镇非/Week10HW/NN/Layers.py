# Author: Zhenfei Lu
# Created Date: 6/25/2024
# Version: 1.0
# Email contact: luzhenfei_2017@163.com, zhenfeil@usc.edu

from enum import Enum
import numpy as np

class LayerType(Enum):
    NONE = 0
    INPUT = 1
    FC = 2
    CONV = 3

class OptimizerType(Enum):
    NONE = 0
    SGD = 1
    ADAM = 2

class LayerBase(object):
    def __init__(self):
        self.LayerType = LayerType.NONE
        self.LayerName = ""
        self.LayerID = ""
        pass

class InputLayer(LayerBase):
    def __init__(self, inputNum):
        super().__init__()
        self.layerType = LayerType.INPUT
        self.neuronsNum = inputNum
        self.outputs = np.zeros((self.neuronsNum, 1), np.float)

    def setInput(self, inputs):
        if(inputs.shape == self.outputs.shape):
            self.outputs = inputs
        else:
            print("Fatal error, input num wrong!!!")


class FullyConnectedLayer(LayerBase):
    def __init__(self, neuronsNum, activation, previousLayerNeuronsNum):
        super().__init__()
        self.layerType = LayerType.FC
        self.neuronsNum = neuronsNum
        self.activation = activation
        self.previousLayerNeuronsNum = previousLayerNeuronsNum

        self.linearPartOutputs = np.zeros((self.neuronsNum, 1), np.float)
        self.outputs = np.zeros((self.neuronsNum, 1), np.float)
        self.error = np.zeros((self.neuronsNum, 1), np.float)  # error, delta, residual ...
        self.weights = np.random.uniform(low=-1.0, high=1.0, size=(self.previousLayerNeuronsNum, self.neuronsNum))
        self.bias = np.random.uniform(low=-1.0, high=1.0, size=(self.neuronsNum, 1))
        # self.weights = np.random.normal(0, pow(self.neuronsNum,-0.5), (self.previousLayerNeuronsNum, self.neuronsNum))  # parent2currentLayer
        # self.bias = np.random.normal(0, pow(self.neuronsNum,-0.5), (self.neuronsNum, 1))
        # self.weights = np.random.random((self.previousLayerNeuronsNum, self.neuronsNum))  # parent2currentLayer
        # self.bias = np.random.random((self.neuronsNum, 1))
        # self.weights = np.zeros((self.previousLayerNeuronsNum, self.neuronsNum))  # parent2currentLayer
        # self.bias = np.zeros((self.neuronsNum, 1))
        self.weightsGradients = np.zeros((self.previousLayerNeuronsNum, self.neuronsNum), np.float)
        self.biasGradients = np.zeros((self.neuronsNum, 1), np.float)

        self.weightsGradients_Previous = np.zeros((self.previousLayerNeuronsNum, self.neuronsNum), np.float)  # for adam
        self.biasGradients_Previous = np.zeros((self.neuronsNum, 1), np.float)

        self.weightsGradients_Previous2 = np.zeros((self.previousLayerNeuronsNum, self.neuronsNum), np.float) # for adam
        self.biasGradients_Previous2 = np.zeros((self.neuronsNum, 1), np.float)

        self.t = 1  # for adam

    def updateOutputs(self, previousLayerOutput):
        self.linearPartOutputs = self.weights.T @ previousLayerOutput + self.bias
        self.outputs = self.activation.func(self.linearPartOutputs)

    def updateError(self, nextLayerError, nextLayerWeights, isHiddenLayer: bool, outputsGroundTruth, lossFunction):
        if (isHiddenLayer):
            self.error = (nextLayerWeights @ nextLayerError) * self.activation.funcDerivative(self.outputs)
        else:  # assume here last layer is MSE loss func or Crossentropy+softmax case
            self.error = lossFunction.funcDerivative(self.outputs, outputsGroundTruth) * self.activation.funcDerivative(self.outputs)

    def updateGradients(self, previousLayerOutput):
        self.weightsGradients = self.weightsGradients + previousLayerOutput @ self.error.T
        self.biasGradients = self.biasGradients + self.error

    def clearGradients(self):
        self.weightsGradients = np.zeros((self.previousLayerNeuronsNum, self.neuronsNum), np.float)
        self.biasGradients = np.zeros((self.neuronsNum, 1), np.float)

    def applyGradients4DerivableParams(self, learningRate, batchSize, optimizerType=OptimizerType.ADAM):
        if(optimizerType == OptimizerType.SGD):
            # small batch
            self.weights = self.weights - learningRate * self.weightsGradients / batchSize
            self.bias = self.bias - learningRate * self.biasGradients / batchSize
        elif(optimizerType == OptimizerType.ADAM):
            # adam
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8
            weightsGradients_moment = beta1 * self.weightsGradients_Previous + (1-beta1) * self.weightsGradients
            weightsGradients_velocity = beta2 * self.weightsGradients_Previous2 + (1-beta2) * (self.weightsGradients**2)
            self.weightsGradients_Previous = weightsGradients_moment
            self.weightsGradients_Previous2 = weightsGradients_velocity
            mhat = weightsGradients_moment / (1 - beta1**self.t)
            vhat = weightsGradients_velocity / (1 - beta2 ** self.t)
            weightsGradients_opt = mhat / (np.sqrt(vhat) + epsilon)
            self.weights = self.weights - learningRate * weightsGradients_opt

            biasGradients_moment = beta1 * self.biasGradients_Previous + (1 - beta1) * self.biasGradients
            biasGradients_velocity = beta2 * self.biasGradients_Previous2 + (1 - beta2) * (self.biasGradients ** 2)
            self.biasGradients_Previous = biasGradients_moment
            self.biasGradients_Previous2 = biasGradients_velocity
            mhat = biasGradients_moment / (1 - beta1 ** self.t)
            vhat = biasGradients_velocity / (1 - beta2 ** self.t)
            biasGradients_opt = mhat / (np.sqrt(vhat) + epsilon)
            self.bias = self.bias - learningRate * biasGradients_opt
            self.bias = np.zeros((self.neuronsNum, 1), np.float)
            self.t = self.t + 1


