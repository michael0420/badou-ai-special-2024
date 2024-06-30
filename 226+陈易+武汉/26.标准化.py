import numpy as np
import matplotlib.pyplot as plt


def max_min(x):
    """
    （0~1）归一化
    :param x: 待归一化数据
    :return: 已归一化数据
    """
    x_max = max(x)      # x中的最大值
    x_min = min(x)      # x中的最小值
    x1 = [(float(i) - x_min)/float(x_max - x_min) for i in x]
    return x1

def max_mean(x):
    """
    (-1~1) 归一化
    :param x: 待归一化数据
    :return: 已归一化数据
    """
    x_max = max(x)      # x中的最大值
    x_min = min(x)      # x中的最小值
    x_mean = np.mean(x) # x的平均值
    x1 = [(float(i) - x_mean)/float(x_max - x_min) for i in x]
    return x1

def z_score(x):
    """
    零均值归一化 ：x∗=(x−μ)/σ 。μ是样本的均值， σ是样本的标准差
    :param x: 待归一化数据
    :return: 已归一化数据
    """
    x_mean = np.mean(x) # x的平均值
    # 求标准差：实际值与均值的差的平方和，除以个数
    x_BZC = sum([(i - x_mean)**2 for i in x]) / len(x)
    # 归一化处理
    x1 = [(i - x_mean)/x_BZC for i in x]
    return x1

def draw(x):
    count_sum = []
    for i in x:
        count = x.count(i)          # 每个元素的个数
        count_sum.append(count)     # 记录每个元素的个数
    return count_sum

if __name__ == '__main__':
    x = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9,
         10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12,
         12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
    a = max_min(x)
    b = max_mean(x)
    c = z_score(x)
    print("0~1归一化：\n",a)
    print("-1~1归一化：\n",b)
    print("零均值归一化：\n",c)

    count_sum = draw(x)
    """"
    绘图 plt.plot(x, y, fmt, **kwargs)： 
    x：表示X轴上的数据点，通常是一个列表、数组或一维序列，用于指定数据点的水平位置
    y：表示Y轴上的数据点，通常也是一个列表、数组或一维序列，用于指定数据点的垂直位置
    fmt：是一个可选的格式字符串，用于指定线条的样式、标记和颜色。例如，‘ro-’ 表示红色圆点线条
    **kwargs：是一系列可选参数，用于进一步自定义线条的属性，如线宽、标记大小、标签等
    """
    plt.plot(x, count_sum)
    plt.plot(a, count_sum)
    plt.plot(b, count_sum)
    plt.plot(c, count_sum)
    plt.show()