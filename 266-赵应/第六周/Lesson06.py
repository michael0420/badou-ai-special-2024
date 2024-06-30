import matplotlib.pyplot as plt
import numpy as np
import random


class KMeans:
    def __init__(self, cluster=2, epochs=10):
        """"
        cluster: 分类数
        epochs：迭代次数
        _centers：中心点集合
        _centers_data: 中心点对应的数据分类
        """
        self._cluster = cluster
        self._epochs = epochs
        self._centers = {}
        self.centers_data = {}

    def calculate(self, data):
        """计算K个分类"""
        # 选取k个中心
        for i in range(self._cluster):
            index = random.randint(0, data.shape[0]-1)
            self._centers[i] = data[index]

        # 开始进行迭代
        for j in range(self._epochs):
            # 初始化中心点分类集合
            for k in range(self._cluster):
                self.centers_data[k] = []
            # 计算数据到每个中心点的距离
            for feature in data:
                # 计算距离并选取距离最小的归为对应类
                distances = []
                for center in self._centers:
                    distances.append(np.linalg.norm(feature - self._centers[center]))
                classify_index = distances.index(min(distances))
                self.centers_data[classify_index].append(feature)


if __name__ == '__main__':
    """
    第一部分：数据集
    X表示二维矩阵数据，篮球运动员比赛数据
    总共20行，每行两列数据
    第一列表示球员每分钟助攻数：assists_per_minute
    第二列表示球员每分钟得分数：points_per_minute
    """
    X = [[0.0888, 0.5885],
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
         [0.1956, 0.4280]]
    data = np.array(X)
    k_means = KMeans(2, epochs=100)
    k_means.calculate(data)
    result = k_means.centers_data
    print(result)
    for i in result.keys():
        dot_dataSource = np.array(result[i])
        plt.scatter(dot_dataSource[:, 0], dot_dataSource[:, 1], marker='*', s=150)
    plt.show(block=True)
