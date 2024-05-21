import random
import numpy as np


def kmeans(X, K):
    m = X.shape[0]  # 列数
    # 从数组中随机选择两个数作为质心
    zhixin1 = X[np.random.randint(0, m, K)]
    # print(zhixin1.shape[1])
    array_1 = None
    for i in range(100):
        # 为每个样本分配到最近的聚类中心
        cluster_assignment = np.zeros(m)
        for r1 in range(m):
            xx = np.sum((X[r1] - zhixin1) ** 2, axis=1)
            cluster_assignment[r1] = np.argmin(xx)
        # 更新聚类中心的位置
        for r2 in range(zhixin1.shape[0]):
            cc = X[cluster_assignment == r2]
            zhixin1[r2] = np.mean(cc, axis=0)
        # 如果聚类中心位置没有变化，则提前终止迭代
        if array_1 is not None and np.all(zhixin1 == array_1):
            break
        array_1 = zhixin1.copy()
    return zhixin1, cluster_assignment


X = np.array([[0.0888, 0.5885],
              [0.1399, 0.8291],
              [0.0747, 0.4974],
              [0.0983, 0.5772],
              [0.1276, 0.5703],
              [0.1671, 0.5835],
              [0.1306, 0.5276],
              [0.1061, 0.5523],
              [0.2446, 0.4007],
              [0.1670, 0.4770],
              [0.2485, 0.4313],
              [0.1227, 0.4909],
              [0.1240, 0.5668],
              [0.1461, 0.5113],
              [0.2315, 0.3788],
              [0.0494, 0.5590],
              [0.1107, 0.4799],
              [0.1121, 0.5735],
              [0.1007, 0.6318],
              [0.2567, 0.4326],
              [0.1956, 0.4280]
              ])
K = 2
zhixin1, cluster_assignment = kmeans(X, K)

print(zhixin1)
print(cluster_assignment)
