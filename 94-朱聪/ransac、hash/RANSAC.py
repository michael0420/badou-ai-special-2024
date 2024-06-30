import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pylab

def gls(X, Y):
    # 构建X,Y矩阵
    X = np.array([[1, i[0]] for i in X])
    Y = np.array([i[0] for i in Y])

    # 求B
    b, k = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)

    return k, b


def ransac():
    """
        np.random.normal(loc=0, scale=1.0, size=None) 默认生成一个服从均值为0，标准差为1的正太分布的随机数

    """

    # 1、在数据中随机选择几个点设定为内群
    # 先准备一堆数据 500个数据点，包含内群和离群点
    # 先随机整一条直线，让500个点都满足该线性，然后添加一些随机数，接着把其中100个点变成离群点，就构造出了一组我们需要的数据集

    X = 20 * np.random.random((500, 1))  # 没有*20得到的都是0-1之间的数
    # 整一条线性直线 y = kx
    real_k = 60 * np.random.normal(size=(1, 1)) # 随机生成一个斜率k
    # real_b = 60 * np.random.normal(size=(1, 1))
    Y = np.dot(X, real_k)

    # 得到了完全分布在该直线上的点集，现在开始添加一些随机数，使其稍微分散开
    X_noisy = X + np.random.normal(size=X.shape) # 生成包含 500行1列的二维数组，值都是满足高斯分布的随机数
    Y_noisy = Y + np.random.normal(size=Y.shape)

    # 接着找100个点，把它们变为离群点，增加干扰
    all_idxs = np.arange(X_noisy.shape[0]) # 生成0-499的索引
    outlier_idxs = np.random.choice(all_idxs, size=100, replace=False) # 随机取100个作为离群点索引

    X_noisy[outlier_idxs] = 20 * np.random.random((100, 1)) # 加入噪声和局外点的Xi
    Y_noisy[outlier_idxs] = 50 * np.random.normal(size=(100, 1))    # 加入噪声和局外点的Yi

    # 至此，准备工作完成了。后续将基于这500个点，来进行RANSAC
    # 先使用最小二乘法计算，后面会将各种方式的情况放在一起对比
    gls_k, gls_b = gls(X_noisy, Y_noisy)

    """
    整体步骤：

    1、在数据中随机选择几个点设定为内群 
    
    2、计算适合内群的模型 e.g. hi=ax+b ->hi=2x+3 。因为我们假定了内群点，所以算出以这些点拟合出的的一条直线（最小二乘法算出来的），当然这个结果大概率不会是最后的结果，因为只取了一部分点
    
    3、把其它刚才没选到的点(x,y)带入刚才建立的模型中，计算是否为内群 e.g. hi=2xi+3->ri ,即会产生误差，因为代入模型得到的结果不一定就是y值。然后会设定一个阈值ri，如果误差小于ri，就认为它是一个内群点
    
    4、记下上边步骤得到的内群数量 
    
    5、重复以上步骤 （1~3），每次重复都能得到一个内群数量
    
    6、比较哪次计算中内群数量最多,内群最多的那次所建的模型就是我们所要求的解
    """

    iterations = 0
    # 初始化一些值
    best_errs = -np.inf
    best_k = 0
    best_b = 0
    while iterations < 1000: # 迭代一定次数
        # 1、随机取若干点作为内群，50个点
        random_idx = np.random.choice(all_idxs, size=50, replace=False)
        # 2、得到50个点后，根据这50个点，使用最小二乘法计算出一个线性结果
        k, b = gls(X_noisy[random_idx], Y_noisy[random_idx])

        # 3、根据y=kx+b，将其他的点代入，计算残差和，看是否满足误差，满足就记作一个内群
        # 有点问题 TODO
        need_idx = np.setdiff1d(all_idxs, random_idx)
        better_errs = 0
        for i in need_idx:
            if ((X_noisy[i] * k + b) - Y_noisy[i]) ** 2 < 7e3:
                better_errs += 1

        if better_errs > 350: # 宁缺毋滥，希望最后找到的内群点数是大于300个
            if 50 + better_errs > best_errs:
                best_errs = better_errs
                best_k = k
                best_b = b

        iterations += 1

    print(best_k, best_b)

    sort_idxs = np.argsort(X[:, 0])
    X_col0_sorted = X[sort_idxs]  # 秩为2的数组

    pylab.plot(X_noisy[:, 0], Y_noisy[:, 0], 'k.', label='data')  # 所有点的散点图
    # 将最开始构造的点集合中的内群点特殊标记出来
    ransac_idx = np.setdiff1d(all_idxs, outlier_idxs)
    pylab.plot(X_noisy[ransac_idx, 0], Y_noisy[ransac_idx, 0], 'bx', label="RANSAC data")

    pylab.plot(X_col0_sorted[:, 0],
               np.dot(X_col0_sorted, real_k)[:, 0],
               label='exact system')

    pylab.plot(X_col0_sorted[:, 0],
               (np.dot(X_col0_sorted, best_k) + best_b)[:, 0],
               label='RANSAC fit')

    pylab.plot(X_col0_sorted[:, 0],
               (np.dot(X_col0_sorted, gls_k) + gls_b)[:, 0],
               label='linear fit')
    pylab.legend()
    pylab.show()

ransac()