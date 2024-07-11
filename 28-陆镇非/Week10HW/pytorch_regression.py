# Author: Zhenfei Lu
# Created Date: 6/25/2024
# Version: 1.0
# Email contact: luzhenfei_2017@163.com, zhenfeil@usc.edu

import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt


class PytorchReressionModel(nn.Module):
    def __init__(self, inputDim, hiddenLayerSize, outputDim):
        super().__init__()

        self.linear1 = nn.Linear(inputDim, hiddenLayerSize)
        self.linear2 = nn.Linear(hiddenLayerSize, hiddenLayerSize)
        self.linear3 = nn.Linear(hiddenLayerSize, hiddenLayerSize)
        self.outputLayer = nn.Linear(hiddenLayerSize, outputDim)
        self.loss = nn.MSELoss()

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        x = self.outputLayer(x)
        return x

    def calLoss(self, y_predict, y):
        return self.loss(y_predict, y)

    def data_iter(self, batch_size, features, labels):
        num_examples = len(features)
        indices = list(range(num_examples))
        random.shuffle(indices)
        for i in range(0, num_examples, batch_size):
            batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
            yield features[batch_indices], labels[batch_indices]

    def fit(self, x, y, epoch, learning_rate, batch_size):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        metric = dict()
        metric["avg_loss"] = list()
        metric["accuracy"] = list()
        for i in range(1, epoch+1):
            self.train()
            for X, Y in self.data_iter(batch_size, x, y):
                optimizer.zero_grad()
                loss = self.calLoss(self.forward(X), Y)
                loss.backward()
                optimizer.step()
            avg_loss = self.calLoss(self.forward(x), y)
            # acc = self.calAccuracy(x, y)
            acc = 1
            metric["avg_loss"].append(avg_loss.item())
            metric["accuracy"].append(acc)
            if i % 1 == 0 or i == epoch:
                print(f"Epoch: {i}, AvgLoss: {avg_loss}, Accuracy: {acc}")
        return metric

    def calAccuracy(self, x, y, showEachResult:bool=False, sentences:list=None):
        acc = 0
        correct = 0
        N = y.shape[0]
        self.eval()
        with torch.no_grad():  # not calculate auto-grad when doing fwd
            for i in range(0,N):
                y_predict = self.forward(x[i].unsqueeze(0))  # shape (1,sentenceLen,2)
                # print(y_predict.shape)
                y_predict_index = (y_predict.argmax(dim=2)).squeeze(0)
                # print(y_predict_index.shape)
                # print(y[i].shape)
                y_true_index = y[i]
                # print(y_predict_index, y_true_index)

                indices = torch.where(y_true_index == -100)[0]  # find the first -100
                if indices.numel() > 0:
                    first_index = indices[0].item()
                    y_predict_index = y_predict_index[0:(first_index)]
                    y_true_index = y_true_index[0:(first_index)]

                if(torch.equal(y_predict_index, y_true_index)):
                    correct = correct + 1
                    if showEachResult:
                        sentence = sentences[i]
                        print(f"y truth: {y_true_index}, y predict: {y_predict_index}, result: {'Correct'}")
                        for j in range(0, y_predict_index.shape[0]):
                            if(y_predict_index[j] == 1):
                                print(sentence[j], end=" / ")
                            elif(y_predict_index[j] == 0):
                                print(sentence[j], end="")
                        print()
                else:
                    if showEachResult:
                        sentence = sentences[i]
                        print(f"y truth: {y_true_index}, y predict: {y_predict_index}, result: {'Wrong'}")
                        for j in range(0, y_predict_index.shape[0]):
                            if (y_predict_index[j] == 1):
                                print(sentence[j], end=" / ")
                            elif (y_predict_index[j] == 0):
                                print(sentence[j], end="")
                        print()
        acc = correct / N
        return acc

    def plotMetric(self, metric, showImmediately):
        plt.figure()
        i = 1
        N = len(metric)
        for key, value in metric.items():
            plt.subplot(1, N, i)
            plt.plot(np.linspace(1,len(value),len(value)), value)
            plt.title(key)
            plt.xlabel('epoch')
            # plt.ylabel('value')
            i = i + 1
        if (showImmediately):
            plt.show()

    def saveModelWeights(self, localPath):
        torch.save(self.state_dict(), localPath)  # touch static method

    def loadModelWeights(self, localPath):
        self.load_state_dict(torch.load(localPath))  # load weights  .path, ***model structure must be same


if __name__ == "__main__":
    X = np.linspace(-5, 5, 100).reshape(-1,1)
    Y = 1*np.sin(X)
    # Y = -5*X + 3

    model = PytorchReressionModel(1, 64, 1)
    hist = model.fit(x=torch.FloatTensor(X), y=torch.FloatTensor(Y), epoch=3000, learning_rate=0.001, batch_size=100)

    plt.figure()
    plt.plot(X, model.forward(torch.FloatTensor(X)).detach().numpy())
    plt.plot(X, Y)
    plt.show()
