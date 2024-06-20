import numpy as np
import matplotlib.pyplot as plt


def zero_to_one(X):
    result = []
    for i in X:
        result.append((i - min(X)) / (max(X) - min(X)))
    return result


def z_core(X):
    "x∗=(x−μ) / σ"
    # 平均值
    avg = np.mean(X)
    # 标准差
    biao_zhun_cha = []
    for i in X:
        biao_zhun_cha.append((i - avg) ** 2)
    σ = sum(biao_zhun_cha) / len(X)
    result = []
    for i in X:
        result.append((i - avg) / σ)
    return result

l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,
     11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
z = z_core(l)
print(zero_to_one(l))
print(z)
# 画图
y_list = []
for i in l:
    y_list.append(l.count(i))


'''
蓝线为原始数据，橙线为z
'''
plt.plot(l, y_list)
plt.plot(z, y_list)
plt.show()
