# -*- coding: utf-8 -*-

import numpy as np  # 导入numpy库，用于进行数值计算和处理数组
import matplotlib.pyplot as plt  # 导入matplotlib库的pyplot模块，用于绘制图形
import random  # 导入random库，用于生成随机数

# 定义生成数据集的参数
SIZE = 500  # 数据点的总数
OUT = 230  # 数据的上限
X = np.linspace(0, 100, SIZE)  # 生成从0到100，共SIZE个数据点的等差数列

Y = []  # 创建一个空列表，用于存储所有的数据值

# 对于X中的每一个元素，执行以下操作
for i in X:
    # 生成一个0到10之间的随机整数，如果这个数大于5，执行下面的if语句，否则执行else语句
    if random.randint(0, 10) > 5:
        # 从0到OUT之间随机生成一个整数，并添加到Y列表中
        Y.append(random.randint(0, OUT))
    else:
        # 再次生成一个0到10之间的随机整数，如果这个数大于5，执行下面的if语句，否则执行else语句
        if random.randint(0, 10) > 5:
            # 根据当前元素i和随机生成的数值计算出一个新的y值，并添加到Y列表中
            Y.append(3 * i + 10 + 3 * random.random())
        else:
            Y.append(3 * i + 10 - 3 * random.random())  # 同上，只是计算公式略有不同

list_x = np.array(X)  # 将X转换为numpy数组，方便后续的数据处理和计算
list_y = np.array(Y)  # 将Y转换为numpy数组，方便后续的数据处理和计算

# 使用matplotlib库绘制原始数据点的散点图
plt.scatter(list_x, list_y)  # 在二维平面上绘制原始数据点，使用散点图展示
plt.show()  # 显示绘制的图形


def linear_regression(list_x, list_y):
    # 进行迭代操作，寻找最佳的线性回归模型参数a和b
    iters = 10000  # 迭代次数
    epsilon = 3  # 内点的误差阈值
    threshold = (SIZE - OUT) / SIZE + 0.01  # 阈值，用于控制早停（early stopping）策略
    best_a, best_b = 0, 0  # 最佳线性回归模型的参数，初始值为0
    pre_total = 0  # 内点数量的初始值，初始为0

    # 进行迭代操作，寻找最佳的线性回归模型参数a和b
    for i in range(iters):
        # 从SIZE个数据点中随机选择两个点，索引存储在sample_index中
        sample_index = random.sample(range(SIZE), 2)
        x_1 = list_x[sample_index[0]]  # 获取第一个点的x值
        x_2 = list_x[sample_index[1]]  # 获取第二个点的x值
        y_1 = list_y[sample_index[0]]  # 获取第一个点的y值
        y_2 = list_y[sample_index[1]]  # 获取第二个点的y值

        # 根据两个点的坐标计算出线性回归模型的斜率a和截距b
        a = (y_2 - y_1) / (x_2 - x_1)  # 计算斜率a
        b = y_1 - a * x_1  # 计算截距b
        total_in = 0  # 内点计数器，初始值为0

        # 对于每一个数据点，计算其对应的预测值，并与真实值进行比较，如果误差小于epsilon，则认为此点为内点，计数器加1
        for index in range(SIZE):
            y_estimate = a * list_x[index] + b  # 根据线性回归模型计算出预测值
            if abs(y_estimate - list_y[index]) < epsilon:  # 判断预测值与真实值的误差是否小于epsilon
                total_in += 1  # 如果小于epsilon，则此点为内点，计数器加1

        # 如果当前的内点数量大于之前所有的内点数量，则更新最佳参数a和b，以及内点数量pre_total
        if total_in > pre_total:  # 记录最大内点数与对应的参数
            pre_total = total_in
            best_a = a
            best_b = b

        # 如果当前的内点数量大于设定的阈值所对应的人数，则跳出循环，不再进行迭代
        if total_in > SIZE * threshold:  # 如果当前内点数量大于阈值所设定的人数，则跳出循环
            break  # 跳出循环
    print("迭代{}次,a = {}, b = {}".format(i, best_a, best_b))  # 输出当前迭代的次数，以及对应的线性回归模型参数a和b
    x_line = list_x  # 获取x轴的数据
    y_line = best_a * x_line + best_b  # 根据最佳线性回归模型计算出y轴的数据
    plt.plot(x_line, y_line, c='r')  # 使用matplotlib库绘制出线性回归模型的直线图，并用红色表示
    plt.scatter(list_x, list_y)  # 使用matplotlib库绘制出原始数据的散点图，并用其他颜色表示
    plt.show()  # 显示绘制的图形


linear_regression(list_x, list_y)