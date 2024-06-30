import numpy as np
import matplotlib.pyplot as plt


# # 1、归一化到(0,1)
# def Normalization1(data):
#     # (x-min(data)) / (max(data)-min(data))
#     return [(x - min(data)) / (max(data) - min(data)) for x in data]
#
#
# # 2、归一化到(-1,1)
# def Normalization2(data):
#     # [1,2,3,4,5]
#     # mean=3
#     # (x-np.mean(data)) / (max(data)-np.mean(data))
#     return [(x - np.mean(data)) / (max(data) - np.mean(data)) for x in data]
#
# def z_score(data):
#     # 先计算均值mean
#     mean=np.mean(data)
#     # 再计算方差σ
#     σ = np.sum([(x-np.mean(data))**2 for x in data])/len(data)
#     return [(x-mean)/σ for x in data]
#
# data=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,
#      11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
# l1=[]
# cs=[]
# for i in data:
#     c=data.count(i)
#     cs.append(c)
# n=Normalization2(data)
# z=z_score(data)
# plt.plot(data,cs)
# plt.plot(z,cs)
# plt.show()

# 标准化为 (0,1)
def Normalization1(arr):
    return [(x - np.min(arr)) / (np.max(arr) - np.min(arr)) for x in arr]


# 标准化为 (-1,1)
def Normalization2(arr):
    return [(x - np.mean(arr)) / (np.max(arr) - np.mean(arr)) for x in arr]


#
def z_score(arr):
    s2 = sum([(x - np.mean(arr)) ** 2 for x in arr]) / len(arr)
    return [(x - np.mean(arr)) / s2 for x in arr]


# if __name__ == '__main__':
#     x = np.array([1, 2, 3, 4, 5, 5, 6, 7, 8, 9])
#     ret = Normalization1(x)
#     ret1=Normalization2(x)
#     ret2=z_score(x)
#     print(ret)
#     print(ret1)
#     print(ret2)

if __name__ == '__main__':
    l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11,
         11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
    cs = []
    for i in l:
        c = l.count(i)
        cs.append(c)
    z = z_score(l)
    plt.plot(l, cs)
    plt.plot(z, cs)
    plt.show()
