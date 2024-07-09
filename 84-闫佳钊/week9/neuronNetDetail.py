import numpy as np


class NeuronNet():
    def __init__(self):
        # 模型结构：输入层2个节点，隐藏层2个节点，输出层2个节点
        self.inpu = [0.05, 0.1]
        self.w = [[0.15, 0.20, 0.25, 0.30],
                  [0.40, 0.45, 0.50, 0.55]]
        self.b = [0.35, 0.60]
        self.outpu = [0.01, 0.99]
        # 学习率r
        self.r = 0.5

    def sigmod(self, x):
        return 1 / (1 + np.exp(-x))

    # zh1,ah1,zo1,ao1
    def neuronNetDetail(self):
        # print(self.w)
        ah = [0, 0]
        zh1 = self.w[0][0] * self.inpu[0] + self.w[0][1] * self.inpu[1] + self.b[0]
        ah[0] = self.sigmod(zh1)
        zh2 = self.w[0][2] * self.inpu[0] + self.w[0][3] * self.inpu[1] + self.b[0]
        ah[1] = self.sigmod(zh2)

        ao = [0, 0]
        zo1 = self.w[1][0] * ah[0] + self.w[1][1] * ah[1] + self.b[1]
        ao[0] = self.sigmod(zo1)
        zo2 = self.w[1][2] * ah[0] + self.w[1][3] * ah[1] + self.b[1]
        ao[1] = self.sigmod(zo2)

        w1 = np.zeros((2, 4))
        # 隐含层到输出层权重跟新
        rangw5 = -(self.outpu[0] - ao[0]) * ao[0] * (1 - ao[0]) * ah[0]
        w1[1][0] = self.w[1][0] - self.r * rangw5
        rangw6 = -(self.outpu[0] - ao[0]) * ao[0] * (1 - ao[0]) * ah[1]
        w1[1][1] = self.w[1][1] - self.r * rangw6
        rangw7 = -(self.outpu[1] - ao[1]) * ao[1] * (1 - ao[1]) * ah[0]
        w1[1][2] = self.w[1][2] - self.r * rangw7
        rangw8 = -(self.outpu[1] - ao[1]) * ao[1] * (1 - ao[1]) * ah[1]
        w1[1][3] = self.w[1][3] - self.r * rangw8
        # 输入层到隐含层权重跟新
        rangah1 = -(self.outpu[0] - ao[0]) * ao[0] * (1 - ao[0]) * self.w[1][0] \
                  + (-(self.outpu[1] - ao[1]) * ao[1] * (1 - ao[1]) * self.w[1][2])
        rangw1 = rangah1 * ah[0] * (1 - ah[0]) * self.inpu[0]
        w1[0][0] = self.w[0][0] - self.r * rangw1
        rangw2 = rangah1 * ah[0] * (1 - ah[0]) * self.inpu[1]
        w1[0][1] = self.w[0][1] - self.r * rangw2
        rangah2 = -(self.outpu[0] - ao[0]) * ao[0] * (1 - ao[0]) * self.w[1][1] \
                  + (-(self.outpu[1] - ao[1]) * ao[1] * (1 - ao[1]) * self.w[1][3])
        rangw3 = rangah2 * ah[1] * (1 - ah[1]) * self.inpu[0]
        w1[0][2] = self.w[0][2] - self.r * rangw3
        rangw4 = rangah2 * ah[1] * (1 - ah[1]) * self.inpu[1]
        w1[0][3] = self.w[0][3] - self.r * rangw4
        self.w = w1
        # print(self.w)
        eTotal = sum((self.outpu[i] - ao[i]) * (self.outpu[i] - ao[i]) for i in range(len(self.outpu))) / len(
            self.outpu)
        print('eTotal={0},ao={1}'.format(eTotal, ao))


if __name__ == '__main__':
    instance = NeuronNet()
    for i in range(10000):
        instance.neuronNetDetail()
