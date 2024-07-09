import numpy as np
import matplotlib.pyplot as plt

# 归一化
def Normalization1(x):
    return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]

def Normalization2(x):
    return [(float(i)-np.mean(x))/(max(x)-min(x)) for i in x]

# 标准化
def z_score(x):
    '''x∗=(x−μ)/σ'''
    x_mean = np.mean(x)  # 计算均值
    s2 = sum([(i - np.mean(x)) * (i - np.mean(x)) for i in x]) / len(x)  # 计算方差
    return [(i - x_mean) / s2 for i in x]  # 返回标准化后的值

l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
cs = [l.count(i) for i in l]
print(cs)

n = Normalization2(l)
z = z_score(l)
print(n)
print(z)

# 绘图
'''
蓝线为原始数据，橙线为标准化数据
'''
plt.plot(l, cs, label='origin data')
plt.plot(n, cs, label='normalization data')
plt.plot(z, cs, label='standardization data')
plt.legend()  # 添加图例
plt.xlabel('data value')  # 添加X轴标签
plt.ylabel('count')  # 添加Y轴标签
plt.title('Data Normalization Example')  # 添加标题
plt.show()
