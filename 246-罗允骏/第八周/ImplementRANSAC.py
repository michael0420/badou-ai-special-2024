# @Time : 2022/11/7 20:11
# @Author : xiao cong
# @Function : RANSAC 算法实现直线拟合

import numpy as np
import matplotlib.pyplot as plt
import random


ITERS = 1000            # 最大迭代次数
SIZE = 50               # 样本数量
RATIO = 0.6             # 期望为内点的比例
INLIERS = SIZE * RATIO  # 内点

# 生成样本数据
X = np.linspace(0, 5, SIZE)
Y = 2 * X + 5
for index in range(SIZE):
    sigma = np.random.uniform(-0.5, 0.5)  # 生成高斯噪声
    Y[index] += sigma


# 绘散点图
plt.figure()
plt.scatter(X, Y)
plt.xlabel("x")
plt.ylabel("y")

# 使用 RANSAC 算法估算模型
iter = 0  # 迭代次数
max_inliers = 0  # 先前最多内点数量
best_a = 0  # 最优参数
best_b = 0
error = 0.5  # 允许最小误差

while iter <= ITERS and max_inliers < INLIERS:

    # 随机选取两个点，计算模型参数
    random_index = random.sample(range(0, SIZE), 2)  # 返回索引列表
    x1 = X[random_index[0]]
    y1 = Y[random_index[0]]
    x2 = X[random_index[1]]
    y2 = Y[random_index[1]]

    a = (y2 - y1) / (x2 - x1)  # 斜率
    b = y1 - a * x1  # 截距
    inliers = 0  # 本次内点数量

    # 代入模型，计算内点数量
    for index in range(SIZE):
        y_estimate = a * X[index] + b
        if abs(Y[index] - y_estimate) <= error:
            inliers += 1

    if inliers >= max_inliers:
        best_a = a
        best_b = b
        max_inliers = inliers

    iter += 1


# 画出拟合直线
Y_estimate = best_a * X + best_b
plt.plot(X, Y_estimate, linewidth=2.0, color="r")
text = "best_a: " + str(round(best_a, 2)) + "\nbest_b:  " + str(round(best_b, 2)) + \
       "\nmax_inliers: " + str(int(max_inliers))
plt.text(3, 6, text, fontdict={'size': 10, 'color': 'r'})
plt.title("RANSAC")
plt.show()
