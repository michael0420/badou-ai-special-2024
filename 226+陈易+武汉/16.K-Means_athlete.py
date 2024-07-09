# encoding = UTF-8

from sklearn.cluster import KMeans

data = [[0.0888, 0.5885],
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
    ]

def k_means(data,lei):
    """
    :param data: 数据集
    :param lei: Kmeans分类数
    :return:
    """
    # clf = KMeans(n_clusters=3) 表示类簇数为3，聚成3类数据，clf即赋值为KMeans
    clf = KMeans(n_clusters=lei)
    # y_pred = clf.fit_predict(X) 载入数据集X，并且将聚类的结果赋值给y_pred
    y_pred = clf.fit_predict(data)
    print("clf:",clf)
    print("y_pred:",y_pred)

    return clf,y_pred

import matplotlib.pyplot as plt

def draw(data,y_pred):
    """
    :param data: 数据集
    :return:
    """
    # 获取数据集的第一列和第二列数据 使用for循环获取 n[0]表示X第一列
    x = [n[0] for n in data]
    y = [n[1] for n in data]

    # 绘制散点图，参数：X轴，Y轴；c=y_pred聚类预测结果；marker类型：o表示圆点，*表示星型，x表示点；
    plt.scatter(x,y,c=y_pred,marker='x')

    # 绘制标题
    plt.title("Kmeans-Basketball Data")

    # 绘制X轴和Y轴
    plt.xlabel("assists_per_munite")
    plt.ylabel("points_per_munite")

    # 设置右上角图例
    plt.legend(["A","B","C"])

    # 显示图形
    plt.show()
"""
第一部分：数据集
X表示二维矩阵数据，篮球运动员比赛数据
总共20行，每行两列数据
第一列表示球员每分钟助攻数：assists_per_minute
第二列表示球员每分钟得分数：points_per_minute

第二部分：KMeans聚类
第三部分：可视化绘图
"""
def test(data,lei):
    """
    :param data: 数据集
    :param lei: Kmeans分类的数量
    :return:
    """
    clf,y_pred = k_means(data,lei)
    draw(data,y_pred)

lei = 3
test(data,lei)