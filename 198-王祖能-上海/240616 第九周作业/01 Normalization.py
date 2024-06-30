import numpy as np
import matplotlib.pyplot as plt


def Normalization(list):
    '''0，1归一化: func = (x-xmin)/(xmax-xmin)'''
    # 注意上面这样操作会修改原始列表数据，每次迭代数据都在不断变化
    # for index, value in enumerate(list):
    #     list[index] = (value - min(list))/(max(list) - min(list))
    # return list
    return [(i - min(list)) / (max(list) - min(list)) for i in list]

def Mean_Normalization(list):
    '''均值标准化: func = (x-mean)/(xmax-xmin)'''
    mu = np.mean(list)
    return [(i - mu)/(max(list) - min(list)) for i in list]


def Zero_Mean_Normalization(list):
    mu = np.mean(list)
    s2 = np.sum([(i - mu) ** 2 / len(list) for i in list])
    return [(i - mu) / (s2 ** 0.5) for i in list]


data = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
data_1 = Normalization(data)
data_2 = Mean_Normalization(data.copy())
data_3 = Zero_Mean_Normalization(data.copy())
print('0-1标准化：\n{}\n均值标准化：\n{}\nZ-score标准化：\n{}'.format(data_1, data_2, data_3))

print('检验是否预原始数据相同：\n{}'.format(data))  # 证明过程中原始数据没有被修改

count = []
for i in data:
    c = data.count(i)
    count.append(c)
print(count)
plt.plot(data, count, c='black', label='original data')
plt.plot(data_1, count, c='red', label='normalization')  # 归一化,对称轴不居0
plt.plot(data_2, count, c='green', label='mean normalization')  # 均值标准化
plt.plot(data_3, count, c='blue', label='z_score')  # Z-score标准化
# 也可以传入多组数据plt.plot(x, y1, x, y2, x, y3)
plt.legend(), plt.show()

x = np.arange(0, 2 * np.pi, 1)
y = np.sin(x)
plt.xlim(0, 2 * np.pi)  # 设置x坐标轴范围
plt.plot(x, y)
plt.show()
