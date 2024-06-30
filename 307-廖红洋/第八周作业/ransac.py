# -*- coding: utf-8 -*-
"""
@File    :   ransac.py
@Time    :   2024/06/15 17:38:55
@Author  :   廖红洋 
"""
import numpy as np
import matplotlib.pyplot as plt

# 生成数据部分使用参考代码的模板，进行了一些修改,后续部分为自写


# 最小二乘法线性回归函数
def Linear(x, y):
    z = np.polyfit(x[:, 0], y[:, 0], deg=1)  # 线性拟合，deg为拟合函数次数,返回的z1为多项式系数，这里为一次系数和0次系数
    p = np.poly1d(z)  # 生成直线
    return p


# 生成完全符合线性的理想点
n_samples = 600  # 样本个数
n_inputs = 1  # 输入变量尺寸
n_outputs = 1  # 输出变量尺寸
A_exact = 20 * np.random.random(
    (n_samples, n_inputs)
)  # 随机生成0-20之间的600个数据:行向量，第一个是个数，第二个是维度
perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))  # 随机线性度，即随机生成一个斜率，为一个值
B_exact = np.dot(A_exact, perfect_fit)  # 从x和斜率生成y值

# 加入高斯噪声
A_noisy = A_exact + np.random.normal(size=A_exact.shape)  # 600 * 1行向量,代表Xi
B_noisy = B_exact + np.random.normal(size=B_exact.shape)  # 600 * 1行向量,代表Yi

# 生成随机局外点
n_outliers = 50
idxs = np.arange(A_noisy.shape[0])  # 获取A的索引：从0-599
np.random.shuffle(idxs)  # 打乱索引顺序
out_idxs = idxs[:n_outliers]
A_noisy[out_idxs] = 10 * np.random.random(
    (n_outliers, n_inputs)
) + 10 * np.random.random() * np.random.random((n_outliers, n_inputs))
A_noisy[out_idxs] = 30 * np.random.random(
    (n_outliers, n_inputs)
) + 20 * np.random.random() * np.random.random((n_outliers, n_inputs))

# 当前数据进行最小二乘拟合
B_pred = Linear(A_noisy, B_noisy)(A_noisy)

# 绘制噪声点与线性回归线
plt.plot(A_noisy, B_noisy, "*", label="原始数据点")  # 绘制所有数据点
plt.plot(A_noisy, B_pred, "-", label="原始线性回归")  # 绘制所有点线性回归线

# 进行Ransac
n_random = 20  # 随机点数目
max = n_random
pbest = Linear(A_noisy, B_noisy)
for i in range(10):
    num = n_random
    np.random.shuffle(idxs)
    idxs_random = idxs[:n_random]
    p = Linear(A_noisy[idxs_random], B_noisy[idxs_random])
    for idx in range(n_samples - n_random):
        dis = abs(p(A_noisy[idx + 20, 0]) - B_noisy[idx + 20, 0])
        if dis < 10:
            num += 1
    if num > max:
        max = num
        pbest = p
B_ransac = pbest(A_noisy[:, 0])
plt.plot(A_noisy, B_ransac, "-", label="RANSAC线性回归")  # 绘制所有点线性回归线

plt.rcParams["font.sans-serif"] = ["simHei"]
plt.legend()
plt.show()
