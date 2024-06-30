from RegressionModel import *
import random

class Ransac(object):
    def __init__(self, model, x, y):
        self.model = model
        self.x = x
        self.y = y

    def sufferAndDivideData(self, num_forFit, x, y):
        if(x.shape[0] != y.shape[0]):
            print("Wrong! x.shape[0] != y.shape[0]")
            return None
        N = y.shape[0]
        indexes = np.array(list(range(0, N)))
        random.shuffle(indexes)
        train_indexes = indexes[0:num_forFit]
        validation_indexes = indexes[num_forFit:]
        return (train_indexes, validation_indexes)

    def checkDataL2lossNorm(self, originalData_indexes, L2NormLossThreshold):
        L2Loss_norm = np.sqrt((self.model(self.model.params, self.x[originalData_indexes]) - self.y[originalData_indexes])**2)
        mask_arr = L2Loss_norm < L2NormLossThreshold
        # print(mask_arr.shape[0])
        return mask_arr.squeeze()   # true or false mask

    def fit(self, num_forFit, maxIter, L2NormLossThreshold, num_interiorData):
        N = self.y.shape[0]
        if((N-num_forFit) < num_interiorData):
            print("Wrong! num_interiorData is impossible: (N-num_forFit) < num_interiorData")
            return None
        bestModel_interiorDataIndex_loss = []
        for i in range(0, maxIter):
            (train_indexes, validation_indexes) = self.sufferAndDivideData(num_forFit, self.x, self.y)
            # print(train_indexes.shape[0])
            # print(validation_indexes.shape[0])
            self.model.resetParamsInitialGuess(assignedParams=[], isAssignedParams=False)
            optimizedParams, loss = self.model.fit(self.x[train_indexes], self.y[train_indexes], maxIter)
            mask_arr = self.checkDataL2lossNorm(validation_indexes, L2NormLossThreshold)
            interior_validation_indexes = validation_indexes[mask_arr]
            # print(interior_validation_indexes.shape[0])
            if(interior_validation_indexes.shape[0] >= num_interiorData):
                interior_validation_indexes = np.concatenate((interior_validation_indexes, train_indexes), axis=0)  # consider indexes for train(fit) is also interior points
                if(len(bestModel_interiorDataIndex_loss)!=0):
                    self.model.resetParamsInitialGuess(assignedParams=[], isAssignedParams=False)
                    optimizedParams, loss = self.model.fit(self.x[interior_validation_indexes], self.y[interior_validation_indexes], maxIter)
                    if(loss < bestModel_interiorDataIndex_loss[2]):
                        # wash interior_validation_indexes again
                        interior_validation_indexes = interior_validation_indexes[self.checkDataL2lossNorm(interior_validation_indexes, L2NormLossThreshold)]
                        self.model.resetParamsInitialGuess(assignedParams=[], isAssignedParams=False)
                        optimizedParams, loss = self.model.fit(self.x[interior_validation_indexes], self.y[interior_validation_indexes], maxIter)
                        bestModel_interiorDataIndex_loss = (self.model, interior_validation_indexes, loss)
                else:
                    self.model.resetParamsInitialGuess(assignedParams=[], isAssignedParams=False)
                    optimizedParams, loss = self.model.fit(self.x[interior_validation_indexes], self.y[interior_validation_indexes], maxIter)
                    # wash interior_validation_indexes again
                    interior_validation_indexes = interior_validation_indexes[self.checkDataL2lossNorm(interior_validation_indexes, L2NormLossThreshold)]
                    self.model.resetParamsInitialGuess(assignedParams=[], isAssignedParams=False)
                    optimizedParams, loss = self.model.fit(self.x[interior_validation_indexes], self.y[interior_validation_indexes], maxIter)
                    bestModel_interiorDataIndex_loss = (self.model, interior_validation_indexes, loss)
        if (len(bestModel_interiorDataIndex_loss) == 0):
            print("Did not get any bestmodel or interiorData satisfied your request")
            return None
        return bestModel_interiorDataIndex_loss