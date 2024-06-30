from GeneralLeastSquareSolver import *
import sys
from enum import Enum
from scipy.optimize import least_squares

class ModelType(Enum):
    NONE = 0
    LinearLayer = 1
    FC = 2
    LINEAR = 3
    NONLINEAR = 4

class RegressionModelBase(object):
    def __init__(self):
        self.modelType = ModelType.NONE

    def __call__(self, *args, **kwargs):
        if(len(args)==0):
            return self.forward(**kwargs)
        elif(len(kwargs) == 0):
            return self.forward(*args)
        else:
            return self.forward(*args, **kwargs)

    def forward(self, params, X):
        pass

class LinearLayer(RegressionModelBase):
    def __init__(self, preivousLayerNodeCount, currentLayerNodeCount, previousLayersParamsTotalCount):
        super().__init__()
        self.modelType = ModelType.LinearLayer
        self.preivousLayerNodeCount = preivousLayerNodeCount
        self.currentLayerNodeCount = currentLayerNodeCount
        self.previousLayersParamsTotalCount = previousLayersParamsTotalCount  # used as a index offset to calculate the index position in the params arr
        self.weightParamsCount = self.preivousLayerNodeCount*self.currentLayerNodeCount
        self.biasParamsCount = self.currentLayerNodeCount
        self.currentLayerParamsTotalCount = self.weightParamsCount + self.biasParamsCount
        self.currentLayerParamsAndPreviousLayersParamsTotalCount = self.previousLayersParamsTotalCount + self.weightParamsCount + self.biasParamsCount
        self.weights = np.random.random((self.preivousLayerNodeCount, self.currentLayerNodeCount))
        self.bias = np.random.random((currentLayerNodeCount, 1))

    def forward(self, params, X):   # X shape is (batchSize, inputSize)
        self.weights = params[self.previousLayersParamsTotalCount:self.previousLayersParamsTotalCount+self.weightParamsCount].reshape(self.preivousLayerNodeCount, self.currentLayerNodeCount)
        self.bias = params[self.previousLayersParamsTotalCount+self.weightParamsCount:self.previousLayersParamsTotalCount+self.weightParamsCount+self.biasParamsCount].reshape(-1,1)
        # fwd = W.T @ X + b
        fwd = np.zeros((X.shape[0], self.currentLayerNodeCount))
        for i in range(0, fwd.shape[0]):
            fwd[i, :] = (self.weights.T @ X[i].reshape(-1, 1) + self.bias).squeeze()
        return fwd

    def getParamsFlatten(self):
        return np.concatenate((self.weights.reshape(-1,1), self.bias.reshape(-1,1)), axis=0)

    def getParamsIndexOffset(self):
        return self.currentLayerParamsAndPreviousLayersParamsTotalCount

class FullyConnectedModel(RegressionModelBase):
    def __init__(self):
        super().__init__()
        self.modelType = ModelType.FC
        self.linear1 = LinearLayer(1, 3, 0)
        self.linear2 = LinearLayer(3, 3, self.linear1.getParamsIndexOffset())
        self.linear3 = LinearLayer(3, 3, self.linear2.getParamsIndexOffset())
        self.linear4 = LinearLayer(3, 1, self.linear3.getParamsIndexOffset())
        self.activation = np.tanh
        self.params = np.vstack((self.linear1.getParamsFlatten(), self.linear2.getParamsFlatten(), self.linear3.getParamsFlatten(), self.linear4.getParamsFlatten()))   # init guess

    def resetParamsInitialGuess(self, isAssignedParams, assignedParams):
        if(isAssignedParams):
            if(self.params.shape[0] == assignedParams.shape[0]):
                self.params = assignedParams
                return True
        else:
            self.params = np.random.random((self.params.shape[0], 1))
            return True

    def forward(self, params, X):
        X = self.activation(self.linear1(params, X))
        X = self.activation(self.linear2(params, X))
        X = self.activation(self.linear3(params, X))
        X = self.linear4(params, X)
        return X

    def residual(self, params, X, Y, others):
        return (self.forward(params, X) - Y).squeeze()
        # return (self.forward(params, X) - Y)

    def fit(self, X, Y, epoch):
        result = least_squares(self.residual, self.params.squeeze(), args=(X, Y, None), method='trf')
        params_optimized = result.x
        self.params = params_optimized.reshape(-1,1)
        return params_optimized
        # solver = GeneralLstSqrSolver(self.residual, self.params, X, Y, None)
        # (res, optimizedParams, loss) = solver.solve(epoch, lossRequest=(LossType.LOSS_NOT_CHANGE, 0.00001), LstSqrSolveMethod=LstSqrSolveMethod.DLS)
        # self.params = optimizedParams
        # return optimizedParams


class LinearModel(RegressionModelBase):
    def __init__(self):
        super().__init__()
        self.modelType = ModelType.LINEAR
        self.linear1 = LinearLayer(1, 1, 0)
        self.params = self.linear1.getParamsFlatten()  # init guess

    def resetParamsInitialGuess(self, assignedParams, isAssignedParams= False):
        if(isAssignedParams):
            if(self.params.shape[0] == assignedParams.shape[0]):
                self.params = assignedParams
                return True
        else:
            self.params = np.random.random((self.params.shape[0], 1))
            return True

    def forward(self, params, X):
        fwd = self.linear1(params, X)
        # A = np.hstack((np.ones((X.shape[0], 1)), X))  #  be same  AX=b
        # fwd = A @ params
        return fwd

    def residual(self, params, X, Y, others):
        return self(params, X) - Y  # use __call__

    def fit(self, X, Y, epoch):
        solver = GeneralLstSqrSolver(residualFunc=self.residual, derivaiableParams=self.params, X=X, Y=Y, otherNoneDerivaiableParams=None)
        (res, optimizedParams, loss) = solver.solve(epoch, lossRequest=(LossType.L2_NORM, 0.00001), LstSqrSolveMethod=LstSqrSolveMethod.LINEAR_CASE)
        self.params = optimizedParams
        return optimizedParams, loss
