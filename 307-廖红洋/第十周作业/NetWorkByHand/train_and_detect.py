# -*- coding: utf-8 -*-
"""
@File    :   train_and_detect.py
@Time    :   2024/06/30 15:56:19
@Author  :   廖红洋 
"""
from NetWork import NetWork
import numpy as np

# 网络初始化
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
n = NetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open("dataset/mnist_train.csv")
training_data_list = training_data_file.readlines()
training_data_file.close()

epoch = 10
for i in range(epoch):
    for record in training_data_list:
        data = record.split(",")
        inputs = (np.asfarray(data[1:])) / 255.0 * 0.99 + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(data[0])] = 0.99
        n.train(inputs, targets)

test_file = open("dataset/mnist_test.csv")
test_data_list = test_file.readlines()
test_file.close()

raw = test_data_list[1].split(",")
lab = str(raw[0])
dat = (np.asfarray(raw[1:])) / 255.0 * 0.99 + 0.01

str = str(np.argmax(n.detect(dat)))

print(lab + "被识别为" + str)
