# -*- coding: utf-8 -*-
"""
@File    :   init.py
@Time    :   2024/06/30 12:09:13
@Author  :   廖红洋 
"""
import numpy as np
import scipy.special


class NetWork:
    def __init__(self, input_nodes, mid_nodes, outcome_nodes, lr):
        self.input = input_nodes
        self.mid = mid_nodes
        self.outcome = outcome_nodes
        self.lr = lr
        self.wih = np.random.normal(0.0, pow(self.mid, -0.5), (self.mid, self.input))
        self.who = np.random.normal(
            0.0, pow(self.outcome, -0.5), (self.outcome, self.mid)
        )
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, input, labels):
        inputs = np.array(input, ndmin=2).T
        targets = np.array(labels, ndmin=2).T

        # 计算与激活
        middat = np.dot(self.wih, inputs)
        middata = self.activation_function(middat)
        outcom = np.dot(self.who, middata)
        outcome = self.activation_function(outcom)

        # 反向传播
        outcome_diff = targets - outcome
        hidden_diff = np.dot(self.who.T, outcome_diff * outcome * (1 - outcome))
        self.who += self.lr * np.dot(
            (outcome_diff * outcome * (1 - outcome)),
            np.transpose(middata),
        )
        self.wih += self.lr * np.dot(
            (hidden_diff * middata * (1 - middata)),
            np.transpose(inputs),
        )

    def detect(self, input):
        hidden_inputs = np.dot(self.wih, input)
        # 计算中间层经过激活函数后形成的输出信号量
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算最外层接收到的信号量
        final_inputs = np.dot(self.who, hidden_outputs)
        # 计算最外层神经元经过激活函数后输出的信号量
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs
